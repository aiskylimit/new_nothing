"""Render the ImageReward grid for the market prompt (3 rows x 5 cols)."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "imagereward_market"

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.style.use(str(REPO / "assets" / "paper.mplstyle"))

COLS = [
    ("Unguided", "unguided"),
    ("Plugin (GNS 50)", "gns50"),
    ("Second-order (GNS 50)", "2nd_order_gns50"),
    (r"Plugin ($k=8$)", "gns50_k8"),
    ("Second-order (raw)", "2nd_order_unnorm"),
]
ROWS = [0, 1, 2]
PROMPT = (
    "Generation: dull, muted, desaturated Indian outdoor market;  "
    "Reward: vibrant Indian outdoor market with colorful stalls and produce"
)


def completed_run(cond_dir: Path):
    runs = [path.parent for path in cond_dir.rglob("rewards.npy")]
    if not runs:
        raise FileNotFoundError(f"No completed run under {cond_dir}")
    return max(runs, key=lambda path: (path / "rewards.npy").stat().st_mtime_ns)


fig, axes = plt.subplots(
    len(ROWS),
    len(COLS),
    figsize=(10, 6.65),
    gridspec_kw={"wspace": 0.02, "hspace": 0.02},
)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.90)
fig.text(
    0.5,
    0.965,
    PROMPT,
    ha="center",
    va="top",
    fontsize=12,
    color="#666666",
    style="italic",
)

for col_idx, (label, key) in enumerate(COLS):
    run = completed_run(DATA / key)
    pngs = sorted(run.glob("[0-9]*.png"))
    rewards = np.load(run / "rewards.npy")
    for row_idx, img_idx in enumerate(ROWS):
        ax = axes[row_idx, col_idx]
        ax.imshow(Image.open(pngs[img_idx]))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if row_idx == 0:
            ax.set_title(f"{label}\nreward={rewards.mean():+.2f}", pad=8)

fig.savefig(HERE / "imagereward_market.pdf")
fig.savefig(HERE / "imagereward_market.png")
print(f"saved -> {HERE}/imagereward_market.{{pdf,png}}")
