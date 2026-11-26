"""Parse the saved stage 2 outputs of the notebook and plot them.

The script reads the notebook as JSON. It does not run any notebook code.
It finds the last code cell whose output has the validation scores, and
it writes:
    results/results.json
    results/val_map_by_checkpoint.png
    results/ap_by_event_tolerance.png
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVENTS = ["challenge", "play", "throwin"]
EVAL_EVERY = 5

# Colors: categorical slots 1-3 of a colorblind-checked palette, light mode.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"challenge": "#2a78d6", "play": "#eb6834", "throwin": "#1baf7a"}


def cell_text(cell):
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream" and out.get("name") == "stdout":
            parts.append("".join(out.get("text", [])))
    return "".join(parts)


def parse_table(lines):
    """Parse the printed pandas Series of (event, tolerance) -> value."""
    table = {}
    event = None
    for line in lines:
        m = re.match(r"^(\w+)?\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if not m:
            continue
        if m.group(1):
            event = m.group(1)
        table.setdefault(event, {})[m.group(2)] = float(m.group(3))
    return table


def parse_notebook(path):
    nb = json.load(open(path))
    text = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "*" * 50 in cell_text(cell):
            text = cell_text(cell)
    if text is None:
        raise SystemExit("no cell with validation scores found")

    lines = text.splitlines()
    checkpoints = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "*" * 50 and i + 2 < len(lines) and lines[i + 2].strip() == "*" * 50:
            score = float(lines[i + 1])
            # The AP table follows the score block and ends at the next "epoch" line.
            j = i + 3
            block = []
            while j < len(lines) and not lines[j].startswith("epoch") and not lines[j].startswith("recall"):
                block.append(lines[j])
                j += 1
            checkpoints.append({"score": score, "ap": parse_table(block)})
            i = j
        else:
            i += 1

    for k, cp in enumerate(checkpoints):
        cp["epoch"] = (k + 1) * EVAL_EVERY
    best = max(range(len(checkpoints)), key=lambda k: checkpoints[k]["score"])
    return {
        "source": "notebooks/BTP_Model.ipynb, saved output of the stage 2 training cell",
        "metric": "event-detection mAP (DFL competition metric), mean over 3 events x 5 tolerances",
        "stage": "stage 2 (1D CNN), 30 epochs, validated after every 5th epoch",
        "checkpoints": [{"epoch": cp["epoch"], "score": round(cp["score"], 4)} for cp in checkpoints],
        "best": {
            "epoch": checkpoints[best]["epoch"],
            "score": round(checkpoints[best]["score"], 4),
            "ap": {ev: {tol: round(v, 4) for tol, v in checkpoints[best]["ap"][ev].items()} for ev in EVENTS},
        },
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_checkpoints(res, out):
    epochs = [c["epoch"] for c in res["checkpoints"]]
    scores = [c["score"] for c in res["checkpoints"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    style(ax)
    ax.plot(epochs, scores, color=COLORS["challenge"], linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    for e, s in zip(epochs, scores):
        ax.annotate(f"{s:.3f}", (e, s), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=TEXT, fontsize=9)
    ax.set_xticks(epochs)
    ax.set_ylim(0.68, 0.72)
    ax.set_xlabel("Epoch", color=TEXT_2)
    ax.set_ylabel("Validation mAP", color=TEXT_2)
    ax.set_title("Stage 2 validation mAP by checkpoint (y-axis starts at 0.68)", color=TEXT, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ap_table(res, out):
    ap = res["best"]["ap"]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax)
    x = 0
    ticks, labels, centers = [], [], []
    for ev in EVENTS:
        xs = []
        for tol, v in ap[ev].items():
            ax.bar(x, v, width=0.8, color=COLORS[ev], edgecolor=SURFACE, linewidth=2)
            ticks.append(x)
            labels.append(f"{float(tol):.2f}")
            xs.append(x)
            x += 1
        centers.append((ev, sum(xs) / len(xs)))
        x += 1  # gap between events
    for ev, c in centers:
        mean = sum(ap[ev].values()) / len(ap[ev])
        ax.text(c, 0.81, f"{ev}\nmean {mean:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("Time tolerance (s)", color=TEXT_2)
    ax.set_ylabel("Average precision", color=TEXT_2)
    ax.set_title(f"AP by event and tolerance, best checkpoint (epoch {res['best']['epoch']})",
                 color=TEXT, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", default=os.path.join(ROOT, "notebooks", "BTP_Model.ipynb"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = parser.parse_args(argv)

    res = parse_notebook(args.notebook)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    plot_checkpoints(res, os.path.join(args.out_dir, "val_map_by_checkpoint.png"))
    plot_ap_table(res, os.path.join(args.out_dir, "ap_by_event_tolerance.png"))
    print(json.dumps(res["checkpoints"]))
    print("best:", res["best"]["epoch"], res["best"]["score"])


if __name__ == "__main__":
    main()
