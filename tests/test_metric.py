"""Tests for event_detection_ap on toy data. They need pandas and numpy only."""
import contextlib
import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dfl.metric import TOLERANCES, event_detection_ap  # noqa: E402


def solution():
    # One video with a scoring interval from 0 s to 100 s and one event of each class.
    return pd.DataFrame({
        "video_id": ["v1"] * 5,
        "time": [0.0, 10.0, 20.0, 30.0, 100.0],
        "event": ["start", "challenge", "play", "throwin", "end"],
    })


def submission(times, scores):
    return pd.DataFrame({
        "video_id": ["v1"] * len(times),
        "time": [float(t) for t in times],
        "event": ["challenge", "play", "throwin"][:len(times)],
        "score": [float(s) for s in scores],
    })


def score(sub):
    # event_detection_ap prints recall and precision tables; keep test output short.
    with contextlib.redirect_stdout(io.StringIO()):
        return event_detection_ap(solution(), sub, TOLERANCES)


def test_perfect_predictions_give_ap_1():
    mean_ap, ap_table = score(submission([10, 20, 30], [0.9, 0.8, 0.7]))
    assert mean_ap == pytest.approx(1.0)
    assert (ap_table == 1.0).all()
    assert len(ap_table) == 15  # 3 events x 5 tolerances


def test_no_predictions_give_ap_0():
    empty = submission([], [])
    mean_ap, ap_table = score(empty)
    assert mean_ap == 0.0
    assert (ap_table == 0.0).all()


def test_predictions_outside_tolerance_give_ap_0():
    # Inside the scoring interval, but more than 0.7 s from every event.
    mean_ap, _ = score(submission([50, 60, 70], [0.9, 0.8, 0.7]))
    assert mean_ap == pytest.approx(0.0)


def test_small_time_error_inside_tolerance_still_matches():
    # 0.1 s error is inside the smallest tolerance of every class.
    mean_ap, _ = score(submission([10.1, 20.1, 30.1], [0.9, 0.8, 0.7]))
    assert mean_ap == pytest.approx(1.0)
