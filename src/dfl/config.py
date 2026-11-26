"""Paths and hyperparameters for both training stages.

Paths come from environment variables. The defaults are relative to the
current directory:

    DFL_DATA_DIR      Kaggle competition data (default: ./data)
    DFL_WORK_DIR      extracted frames, checkpoints, frame scores (default: ./work)
    DFL_TRAIN_CSV     event labels (default: $DFL_DATA_DIR/train.csv)
    DFL_VIDEO_DIR     match videos (default: $DFL_DATA_DIR/train)
    DFL_FRAME_PREDS   stage 1 frame scores, input to stage 2
                      (default: $DFL_WORK_DIR/prob_train_psi.npy.npz)

The hyperparameter values are the values from the original notebook.
"""
import os

# ---------------------------------------------------------------- paths
DATA_DIR = os.environ.get("DFL_DATA_DIR", "./data")
WORK_DIR = os.environ.get("DFL_WORK_DIR", "./work")

TRAIN_CSV = os.environ.get("DFL_TRAIN_CSV", os.path.join(DATA_DIR, "train.csv"))
VIDEO_DIR = os.environ.get("DFL_VIDEO_DIR", os.path.join(DATA_DIR, "train"))

IMG_SOURCE = "img"
IMG_DIR = os.path.join(WORK_DIR, IMG_SOURCE)  # extracted JPEG frames
MODEL_DIR = os.path.join(WORK_DIR, "models")  # stage 1 checkpoints
FRAME_PREDS = os.environ.get(
    "DFL_FRAME_PREDS", os.path.join(WORK_DIR, "prob_train_psi.npy.npz")
)

# The notebook read the frame rate from this video (the result was 25.0).
FPS_PROBE_VIDEO_ID = "1606b0e6_0"

# ---------------------------------------------------------------- data
IMSIZE = [540, 768]  # albumentations Resize(height, width)
IMG_SIZE = (540, 768)  # cv2.resize dsize at frame extraction
BACK_INTERVAL = 20  # sample 1 background frame every 20 frames (train)
BACK_INTERVAL_VAL = 1  # use every background frame (validation)
ERR_TOL = 1  # index into ERR_TOL_SECONDS that sets the label window

# Tolerances in seconds per event. They set the start/end and pre/post
# label windows around each event.
ERR_TOL_SECONDS = {
    "challenge": [0.30, 0.40, 0.50, 0.60, 0.70],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}

# Match halves used in stage 1. The original run used a subset of the
# training games. The other halves were commented out in the notebook.
VIDEO_ID_SPLIT = {
    "val": [
        "35bd9041_0",
    ],
    "train": [
        "1606b0e6_0",
        "ecf251d4_0",
    ],
}

EVENT_NAMES = ["challenge", "throwin", "play"]
EVENT_NAMES_WITH_BACKGROUND = ["background", "challenge", "play", "throwin"]
NUM_CLASSES = 10  # background, 3 events, 3 pre_ classes, 3 post_ classes

# ---------------------------------------------------------------- stage 1
MODEL_NAME = "tf_efficientnet_b0"
USE_AMP = True
BATCH_SIZE = 40
N_EPOCHS = 30
NUM_WORKERS = 2
COSINE = True
INIT_LR = 2e-4
MIXUP = True
KERNEL_TYPE = "{}-{}".format(MODEL_NAME, IMSIZE[0])
FOLD = 0  # only used in checkpoint file names
DEVICE = "cuda"

# Post-processing of frame scores into detections, stage 1 validation.
FRAME_RATE = 25
STAGE1_WINDOW = 2
STAGE1_IGNORE = 16

# ---------------------------------------------------------------- stage 2
STEPS = 25  # the 1D CNN sees 2 * STEPS + 1 frames around each frame
TEMPORAL_EPOCHS = 30
TEMPORAL_LR = 1e-3
TEMPORAL_BATCH_TRAIN = 256
TEMPORAL_BATCH_VAL = 512
TEMPORAL_EVAL_EVERY = 5  # validate after every 5th epoch
# Frame scores from videos whose id contains this string train the 1D CNN.
# Frame scores from all other videos in FRAME_PREDS validate it.
TEMPORAL_TRAIN_VIDEO = "35bd9041"
STAGE2_WINDOW = 1
STAGE2_IGNORE = 16
TEMPORAL_CHECKPOINT = os.path.join(WORK_DIR, "cnn1d_{}_32_psi.pth".format(STEPS))
