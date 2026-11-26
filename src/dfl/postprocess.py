"""Turn per-frame class scores into scored event detections."""
import numpy as np
import pandas as pd

from dfl import config


def make_sub(prob, filenames, window_size, ignore_width,
             frame_rate=config.FRAME_RATE,
             event_names=config.EVENT_NAMES,
             event_names_with_background=config.EVENT_NAMES_WITH_BACKGROUND):
    """Make a detection table (video_id, time, event, score) from frame scores.

    prob: array (n_frames, 4) with the scores of background, challenge,
    play and throwin. filenames: one "<dir>/<video>-<frame>.jpg" per row.
    For each video and event, the scores are smoothed with a centered
    rolling mean of window_size. Then peaks are taken in decreasing score
    order, and the ignore_width frames on each side of a peak are removed.
    """
    df = pd.DataFrame(prob, columns=event_names_with_background)
    df['video_name'] = filenames
    df['video_id'] = df['video_name'].str.split('-').str[0].str.split("/").str[-1]
    df['frame_id'] = df['video_name'].str.split('-').str[1].str.split('.').str[0].astype(int)

    train_df = []
    for video_id, gdf in df.groupby('video_id'):
        for i, event in enumerate(event_names):
            prob_arr = gdf[event].rolling(window=window_size, center=True).mean().fillna(-100).values
            sort_arr = np.argsort(-prob_arr)
            rank_arr = np.empty_like(sort_arr)
            rank_arr[sort_arr] = np.arange(len(sort_arr))
            idx_list = []
            for i in range(len(prob_arr)):
                this_idx = sort_arr[i]
                if this_idx >= 0:
                    idx_list.append(this_idx)
                    for parity in (-1, 1):
                        for j in range(1, ignore_width + 1):
                            ex_idx = this_idx + j * parity
                            if ex_idx >= 0 and ex_idx < len(prob_arr):
                                sort_arr[rank_arr[ex_idx]] = -1
            this_df = gdf.reset_index(drop=True).iloc[idx_list].reset_index(drop=True)
            this_df["score"] = prob_arr[idx_list]
            this_df['event'] = event
            train_df.append(this_df)
    train_df = pd.concat(train_df)
    train_df['time'] = train_df['frame_id'] / frame_rate

    return train_df.reset_index(drop=True)
