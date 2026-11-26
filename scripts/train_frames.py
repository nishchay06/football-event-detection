"""Stage 1: frame extraction, frame classifier training, frame-score export.

Commands:
    extract   write the frames of the split videos as JPEG files
    train     train the EfficientNet-B0 frame classifier
    export    run a checkpoint over all frames of some videos and save the
              frame scores that stage 2 reads (DFL_FRAME_PREDS)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dfl import config  # noqa: E402
from dfl.data import (build_frame_table, dflDataset, extract_images,  # noqa: E402
                      prepare_labels, transforms_train, transforms_val)
from dfl.metric import TOLERANCES, event_detection_ap  # noqa: E402
from dfl.models import net  # noqa: E402
from dfl.postprocess import make_sub  # noqa: E402


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, optimizer, criterion, scaler, device=config.DEVICE):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target, target2, lam, _) in bar:
        data, target, target2, lam = data.to(device), target.to(device).long(), target2.to(device).long(), lam.to(device).float()
        loss_func = criterion
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            logits = model(data).squeeze(1)
            if config.MIXUP:
                loss = mixup_criterion(criterion, logits, target, target2, lam).mean()
            else:
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


def val_epoch(model, loader, criterion, solution, device=config.DEVICE):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    FILES = []

    with torch.no_grad():
        for (data, target, _, _, files) in tqdm(loader):
            data, target = data.to(device), target.to(device).long()
            logits = model(data)
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

    # analyze
    cm = confusion_matrix(TARGETS, PREDS.argmax(1))
    print(cm)
    for i, val in enumerate(cm):
        print("for class {}: accuracy: {}".format(i, val[i] / sum(val) * 100))

    val_df = make_sub(PREDS[:, :4], FILES, config.STAGE1_WINDOW, config.STAGE1_IGNORE)
    score, ap_table = event_detection_ap(
        solution[solution['video_id'].isin(val_df['video_id'].unique())],
        val_df[['video_id', 'time', 'event', 'score']],
        TOLERANCES,
    )
    print(score)
    print(ap_table)

    return val_loss, acc, score, PREDS, FILES, TARGETS


def load_solution():
    return pd.read_csv(config.TRAIN_CSV, usecols=['video_id', 'time', 'event'])


def cmd_extract(args):
    os.makedirs(config.IMG_DIR, exist_ok=True)
    wanted = set(config.VIDEO_ID_SPLIT["train"] + config.VIDEO_ID_SPLIT["val"])
    for video_path in sorted(glob.glob(os.path.join(config.VIDEO_DIR, "*.mp4"))):
        video_id = os.path.basename(video_path).split('.')[0]
        if args.all or video_id in wanted:
            extract_images(video_path, config.IMG_DIR)


def cmd_train(args):
    df = prepare_labels()
    solution = load_solution()
    df_train = build_frame_table(df, config.VIDEO_ID_SPLIT["train"])
    df_val = build_frame_table(df, config.VIDEO_ID_SPLIT["val"], VAL=True)

    device = config.DEVICE
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    # setup dataset
    dataset_train = dflDataset(df_train, transform=transforms_train)
    dataset_valid = dflDataset(df_val, transform=transforms_val, test=True)

    # Setup dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.BATCH_SIZE, sampler=RandomSampler(dataset_train), num_workers=config.NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=config.BATCH_SIZE, sampler=SequentialSampler(dataset_valid), num_workers=config.NUM_WORKERS)

    # Initialize model
    model = net(config.MODEL_NAME)
    model = model.to(device)

    print(len(dataset_train), len(dataset_valid))

    # We use Cosine annealing LR scheduling
    optimizer = optim.Adam(model.parameters(), lr=config.INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.N_EPOCHS)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best = 0
    for epoch in range(1, config.N_EPOCHS + 1):
        torch.cuda.empty_cache()
        print(time.ctime(), 'Epoch:', epoch)

        scheduler.step(epoch - 1)

        train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, acc, score, PREDS, FILES, TARGETS = val_epoch(model, valid_loader, criterion, solution, device)

        if score > best:
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f'{config.KERNEL_TYPE}_epoch{epoch}_fold{config.FOLD}.pth'))
            best = score


def cmd_export(args):
    """Save frame scores in the format that stage 2 reads.

    The exported notebook loads this file but does not contain the code
    that wrote it. This command writes the same keys: arr_0 = frame scores
    (n, 10), arr_1 = file names, arr_2 = class labels. Frames are sorted by
    video and frame number, because stage 2 reads windows of consecutive
    rows.
    """
    df = prepare_labels()
    solution = load_solution()
    video_ids = args.videos or (config.VIDEO_ID_SPLIT["val"] + config.VIDEO_ID_SPLIT["train"])
    df_all = build_frame_table(df, video_ids, VAL=True).sort_values(["video", "frame"])

    device = config.DEVICE
    model = net(config.MODEL_NAME)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    dataset = dflDataset(df_all, transform=transforms_val, test=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=SequentialSampler(dataset), num_workers=config.NUM_WORKERS)
    _, _, _, PREDS, FILES, TARGETS = val_epoch(model, loader, nn.CrossEntropyLoss(), solution, device)

    os.makedirs(os.path.dirname(config.FRAME_PREDS) or ".", exist_ok=True)
    np.savez(config.FRAME_PREDS, PREDS, np.array(FILES), TARGETS)
    print("saved", config.FRAME_PREDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="extract JPEG frames from the videos")
    p.add_argument("--all", action="store_true", help="extract every video in DFL_VIDEO_DIR, not only the split videos")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("train", help="train the frame classifier")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="save frame scores for stage 2")
    p.add_argument("--checkpoint", required=True, help="stage 1 checkpoint (.pth)")
    p.add_argument("--videos", nargs="+", help="video ids (default: all split videos)")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
