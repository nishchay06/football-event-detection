"""Stage 2: train the 1D CNN on the sequence of stage 1 frame scores.

Input: the frame-score file from stage 1 (DFL_FRAME_PREDS). Frame scores
from video TEMPORAL_TRAIN_VIDEO (35bd9041) train the model. Frame scores
from every other video in the file validate it. The script validates after
every 5th epoch and saves the checkpoint with the best score.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dfl import config  # noqa: E402
from dfl.data import linDataset  # noqa: E402
from dfl.metric import TOLERANCES, event_detection_ap  # noqa: E402
from dfl.models import cnn  # noqa: E402
from dfl.postprocess import make_sub  # noqa: E402


def train_epoch(model, loader, optimizer, criterion, scaler, device=config.DEVICE):
    model.train().float()
    train_loss = []
    bar = tqdm(loader)
    for (data, target, _) in bar:
        data, target = data.to(device), target.to(device).long()
        loss_func = criterion
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            logits = model(data).squeeze(1)
            loss = loss_func(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return np.mean(train_loss)


def val_epoch(model, loader, criterion, solution, device=config.DEVICE, get_output=False):
    model.eval().half()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    FILES = []

    with torch.no_grad():
        for (data, target, files) in tqdm(loader):
            data, target = data.to(device), target.to(device).long()
            logits = model(data.half())
            logits = logits.squeeze(1)
            loss_func = criterion
            loss = loss_func(logits, target)

            pred = logits.sigmoid().detach()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)
            FILES.extend(files)

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = np.sum(PREDS.argmax(1) == TARGETS) / len(PREDS.argmax(1)) * 100

    val_df = make_sub(PREDS[:, :4], FILES, config.STAGE2_WINDOW, config.STAGE2_IGNORE)
    score, ap_table = event_detection_ap(
        solution[solution['video_id'].isin(val_df['video_id'].unique())],
        val_df[['video_id', 'time', 'event', 'score']],
        TOLERANCES,
    )
    print("*" * 50)
    print(score)
    print("*" * 50)

    print(ap_table)

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, score


def load_frame_preds(path):
    npz = np.load(path)  # stage 1 inference results
    PREDS = npz["arr_0"]
    FILES = npz["arr_1"]
    TARGETS = npz["arr_2"]
    FILES = np.array(FILES)
    return PREDS, FILES, TARGETS


def split_train_val(FILES):
    """0 = frame of the stage 2 training video, 1 = validation frame."""
    trainval = []
    for f in FILES:
        video = f.split("/")[-1].split("-")[0]
        if config.TEMPORAL_TRAIN_VIDEO in video:
            trainval.append(0)
        else:
            trainval.append(1)
    return np.array(trainval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-preds", default=config.FRAME_PREDS, help="stage 1 frame-score file (.npz)")
    parser.add_argument("--checkpoint", default=config.TEMPORAL_CHECKPOINT, help="where to save the best 1D CNN")
    args = parser.parse_args(argv)

    device = config.DEVICE
    solution = pd.read_csv(config.TRAIN_CSV, usecols=['video_id', 'time', 'event'])
    PREDS, FILES, TARGETS = load_frame_preds(args.frame_preds)
    trainval = split_train_val(FILES)

    dataset_train = linDataset(PREDS[trainval == 0], TARGETS[trainval == 0], FILES[trainval == 0])
    dataset_valid = linDataset(PREDS[trainval == 1], TARGETS[trainval == 1], FILES[trainval == 1])

    # Setup dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.TEMPORAL_BATCH_TRAIN, sampler=RandomSampler(dataset_train), num_workers=config.NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=config.TEMPORAL_BATCH_VAL, sampler=SequentialSampler(dataset_valid), num_workers=config.NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    model = cnn()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.TEMPORAL_LR)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    scores = []
    best = 0
    for i in range(config.TEMPORAL_EPOCHS):
        print("epoch ", i)
        train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        if i % config.TEMPORAL_EVAL_EVERY == config.TEMPORAL_EVAL_EVERY - 1:
            _, _, score = val_epoch(model, valid_loader, criterion, solution, device)
            scores.append(score)
            if score > best:
                best = score
                torch.save(model.state_dict(), args.checkpoint)
    print("validation scores:", scores)


if __name__ == "__main__":
    main()
