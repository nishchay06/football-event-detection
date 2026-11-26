"""Label preparation, frame extraction, datasets and transforms."""
import os

import albumentations
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dfl import config


# ---------------------------------------------------------------- labels
def prepare_labels(train_csv=config.TRAIN_CSV, video_dir=config.VIDEO_DIR):
    """Read train.csv and add the start/end and pre/post label windows.

    For each event, the label window is +/- half of the chosen tolerance
    (start_<event> to end_<event>). The pre_<event> and post_<event> windows
    extend the window by the same amount on each side. The original event
    rows are then dropped. The "start" and "end" rows of the scoring
    intervals stay. Returns the table with a "frame" column.
    """
    err_tol = config.ERR_TOL_SECONDS
    df = pd.read_csv(train_csv)
    additional_events = []
    for arr in df.sort_values(['video_id', 'time', 'event', 'event_attributes']).values:
        if arr[2] in err_tol:
            tol = err_tol[arr[2]][config.ERR_TOL] / 2
            additional_events.append([arr[0], arr[1] - tol, 'start_' + arr[2], arr[3]])
            additional_events.append([arr[0], arr[1] + tol, 'end_' + arr[2], arr[3]])

    for arr in df.sort_values(['video_id', 'time', 'event', 'event_attributes']).values:
        if arr[2] in err_tol:
            tol = err_tol[arr[2]][config.ERR_TOL] / 2
            additional_events.append([arr[0], arr[1] - tol * 2, 'pre_' + arr[2], arr[3]])
            additional_events.append([arr[0], arr[1] + tol * 2, 'post_' + arr[2], arr[3]])

    df = pd.concat([df, pd.DataFrame(additional_events, columns=df.columns)])
    df = df[~df['event'].isin(config.EVENT_NAMES)]
    df = df.sort_values(['video_id', 'time'])

    cap = cv2.VideoCapture(os.path.join(video_dir, config.FPS_PROBE_VIDEO_ID + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    df["frame"] = df["time"] * fps
    return df


# We add pre and post actions to fully utilize the labels.
def get_label(label):
    if label == "background":
        label = 0
    elif label == "challenge":
        label = 1
    elif label == "play":
        label = 2
    elif label == "throwin":
        label = 3
    elif label == "pre_challenge":
        label = 4
    elif label == "pre_play":
        label = 5
    elif label == "pre_throwin":
        label = 6
    elif label == "post_challenge":
        label = 7
    elif label == "post_play":
        label = 8
    elif label == "post_throwin":
        label = 9
    label = np.array(label)
    return label


# ---------------------------------------------------------------- frames
def frame_path(video_id, frame):
    return os.path.join(config.IMG_DIR, video_id + "-" + str(frame).zfill(6) + ".jpg")


def extract_images(video_path, out_dir=config.IMG_DIR):
    """Write every frame of a video as <video>-<frame:06>.jpg (frames start at 1)."""
    video_name = os.path.basename(video_path).split('.')[0]
    cam = cv2.VideoCapture(video_path)
    print(video_path)
    frame_count = 1
    while True:
        successed, img = cam.read()
        if not successed:
            break
        outfile = f'{out_dir}/{video_name}-{frame_count:06}.jpg'
        img = cv2.resize(img, dsize=config.IMG_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(outfile, img)
        frame_count += 1


def get_df(df, video_id, VAL=False):
    """Make the per-frame table (frame, cls, video) for one video.

    Frames inside the label windows get the event or pre/post class. Other
    frames become "background": every BACK_INTERVAL frames for training,
    every BACK_INTERVAL_VAL frames for validation.
    """
    df_video = df[df.video_id == video_id]

    print(video_id, df_video.shape)

    # current status => background, play, challenge, throwin
    arr = df_video[['frame', 'event']].values

    start = None
    data = []
    for a in arr:
        if "pre_" in a[1]:
            start = a[0]
            cls = a[1]
        if "start_" in a[1]:
            data.append({"start": start, "end": a[0], "cls": cls})
            start = a[0]
            cls = a[1].split("_")[-1]
        if "end_" in a[1]:
            end = a[0]
            data.append({"start": start, "end": end, "cls": cls})
        if "post_" in a[1]:
            data.append({"start": end, "end": a[0], "cls": a[1]})
    # make events
    out = []
    for d in data:
        start = int(d["start"])

        if os.path.isfile(frame_path(video_id, start)):
            out.append({"frame": start, "cls": d["cls"], "video": video_id})
        start += 1
        while start <= d["end"]:
            if os.path.isfile(frame_path(video_id, start)):
                out.append({"frame": start, "cls": d["cls"], "video": video_id})
            start += 1

    df2 = pd.DataFrame(out)
    if not VAL:
        for i in range(10, df2.frame.max(), config.BACK_INTERVAL):
            if np.sum(df2.frame.isin([i])) == 0:
                if os.path.isfile(frame_path(video_id, i)):
                    out.append({"frame": i, "cls": "background", "video": video_id})
                else:
                    print("pass:", i)
    else:
        for i in range(10, df2.frame.max(), config.BACK_INTERVAL_VAL):
            if np.sum(df2.frame.isin([i])) == 0:
                if os.path.isfile(frame_path(video_id, i)):
                    out.append({"frame": i, "cls": "background", "video": video_id})
                else:
                    print("pass:", i)
    df2 = pd.DataFrame(out)
    return df2


def build_frame_table(df, video_ids, VAL=False):
    """Concatenate get_df over several videos."""
    for i, video_id in enumerate(video_ids):
        df2 = get_df(df, video_id, VAL)
        if i > 0:
            df_out = pd.concat([df_out, df2])
        else:
            df_out = df2
    return df_out


# ---------------------------------------------------------------- datasets
class dflDataset(Dataset):
    """Stage 1 dataset.

    Each sample stacks 3 grayscale frames (t-1, t, t+1) as the 3 image
    channels. In training (test=False), mixup blends 90% of the samples
    with a random second sample.
    """

    def __init__(self,
                 df,
                 test=False,
                 transform=None,
                 ):

        self.df = df.reset_index(drop=True)
        self.test = test
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        images = []
        for i in [-1, 0, 1]:
            tiff_file = frame_path(row.video, row.frame + i)
            im = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
            images.append(im)
        images = np.array(images).transpose(1, 2, 0)

        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']
        # The file name identifies the frame in post-processing. It is
        # returned as "img/<video>-<frame>.jpg", as in the notebook
        # ("work/img/..."), so that a "-" in DFL_WORK_DIR cannot break the
        # parsing in make_sub.
        file = os.path.join(config.IMG_SOURCE, os.path.basename(tiff_file))
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        # Load labels
        label = get_label(row.cls)

        # Mixup part
        rd = np.random.rand()
        label2 = label
        gamma = np.array(np.ones(1)).astype(np.float32)[0]
        if config.MIXUP and rd < 0.9 and not self.test:
            mix_idx = np.random.randint(0, len(self.df) - 1)
            row2 = self.df.iloc[mix_idx]
            images2 = []
            for i in [-1, 0, 1]:
                tiff_file = frame_path(row2.video, row2.frame + i)
                im = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
                images2.append(im)
            images2 = np.array(images2).transpose(1, 2, 0)

            if self.transform is not None:
                images2 = self.transform(image=images2)['image']

            images2 = images2.astype(np.float32)
            images2 /= 255
            images2 = images2.transpose(2, 0, 1)
            # blend image
            gamma = np.array(np.random.beta(1, 1)).astype(np.float32)
            images = ((images * gamma + images2 * (1 - gamma)))
            # blend labels
            label2 = get_label(row2.cls)

        return torch.tensor(images), torch.tensor(label), torch.tensor(label2), torch.tensor(gamma), file


class linDataset(Dataset):
    """Stage 2 dataset: a window of 2 * STEPS + 1 frame-score vectors.

    The window is centered on frame i. Near the start and the end of the
    array, the center moves inward by STEPS. The label is the class of
    frame i.
    """

    def __init__(self,
                 df,
                 target,
                 files
                 ):

        self.df = df
        self.target = target
        self.files = files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        STEPS = config.STEPS
        if i < STEPS + 2:
            i += STEPS
        if i > len(self.df) - STEPS - 2:
            i -= STEPS
        data = self.df[i - STEPS:i + STEPS + 1]

        # Load labels
        label = self.target[i]

        return torch.tensor(data).transpose(0, 1), torch.tensor(label), self.files[i]


# ---------------------------------------------------------------- transforms
IMSIZE = config.IMSIZE

transforms_train = albumentations.Compose([
    albumentations.ShiftScaleRotate(scale_limit=0.3, rotate_limit=45, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
    ], p=0.9),
    A.Cutout(num_holes=24, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
    albumentations.Rotate(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Resize(IMSIZE[0], IMSIZE[1], p=1.0),
])
transforms_val = albumentations.Compose([albumentations.Resize(IMSIZE[0], IMSIZE[1], p=1.0), ])
