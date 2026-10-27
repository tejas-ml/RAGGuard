"""Generate RAGGuard paper figures."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x, y, w, h, label, fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=16, linewidth=1.5)
    ax.add_patch(arr)


def main_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4)
    ax.axis("off")

    draw_box(ax, 0.25, 2.1, 1.2, 0.8, "User\nquery")
    draw_box(ax, 1.95, 2.1, 1.4, 0.8, "Retriever")
    draw_box(ax, 3.85, 2.1, 1.75, 0.8, "Retrieved\nchunks")
    draw_box(ax, 6.05, 2.0, 1.8, 1.0, "RAGGuard\nprivacy firewall")
    draw_box(ax, 8.35, 2.1, 1.45, 0.8, "LLM")
    draw_box(ax, 8.35, 0.65, 1.45, 0.8, "Safe\nanswer")

    draw_box(ax, 5.7, 0.35, 0.95, 0.55, "public\nsafe", fontsize=8)
    draw_box(ax, 6.85, 0.35, 1.0, 0.55, "redact", fontsize=8)
    draw_box(ax, 8.05, 0.35, 1.0, 0.55, "block", fontsize=8)

    ax.text(6.95, 3.35, "Score = PII + secrets + access level + injection + query risk", ha="center", fontsize=9)
    ax.text(2.65, 1.25, "Synthetic company corpus\n(public, internal, private, poisoned)", ha="center", fontsize=9)

    arrow(ax, 1.45, 2.5, 1.95, 2.5)
    arrow(ax, 3.35, 2.5, 3.85, 2.5)
    arrow(ax, 5.6, 2.5, 6.05, 2.5)
    arrow(ax, 7.85, 2.5, 8.35, 2.5)
    arrow(ax, 9.07, 2.1, 9.07, 1.45)
    arrow(ax, 6.95, 2.0, 6.95, 0.95)

    ax.set_title("RAGGuard: Context-Aware Privacy Firewall for RAG", fontsize=14, pad=10)
    save(fig, out / "main.png")


def panel_a(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    data = summary.set_index("system").loc[["Baseline RAG", "RAGGuard"]]
    values = data["leakage_rate_any_sensitive"] * 100.0
    ax.bar(values.index, values.values)
    ax.set_ylabel("Sensitive queries with leakage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Panel A: Sensitive information leakage")
    for idx, value in enumerate(values.values):
        ax.text(idx, value + 2, f"{value:.1f}%", ha="center")
    save(fig, out / "figure2.png")


def panel_b(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    data = summary.set_index("system").loc[["Baseline RAG", "RAGGuard"]]
    values = data["answer_utility_benign"] * 100.0
    ax.bar(values.index, values.values)
    ax.set_ylabel("Benign answer utility (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Panel B: Utility after filtering")
    for idx, value in enumerate(values.values):
        ax.text(idx, value + 2, f"{value:.1f}%", ha="center")
    save(fig, out / "figure3.png")


def panel_c(sweep: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    x = sweep["threshold"]
    privacy = 100.0 * (1.0 - sweep["leakage_rate_any_sensitive"])
    utility = 100.0 * sweep["answer_utility_benign"]

    line1 = ax1.plot(x, privacy, marker="o", label="Privacy protection")
    ax1.set_xlabel("RAGGuard sensitivity threshold")
    ax1.set_ylabel("Privacy protection (%)")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    line2 = ax2.plot(x, utility, marker="s", linestyle="--", label="Benign utility")
    ax2.set_ylabel("Benign utility (%)")
    ax2.set_ylim(0, 105)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left")
    ax1.set_title("Panel C: Privacy-utility threshold tradeoff")
    save(fig, out / "figure4.png")


def copy_to_paper(fig_dir: Path, paper_out: Path) -> None:
    paper_out.mkdir(parents=True, exist_ok=True)
    for path in fig_dir.glob("*.png"):
        shutil.copy2(path, paper_out / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--paper-out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    summary = pd.read_csv(args.results / "metrics_summary.csv")
    sweep = pd.read_csv(args.results / "threshold_sweep.csv")

    args.out.mkdir(parents=True, exist_ok=True)
    main_architecture(args.out)
    panel_a(summary, args.out)
    panel_b(summary, args.out)
    panel_c(sweep, args.out)
    copy_to_paper(args.out, args.paper_out)
    print(f"Wrote figures to {args.out} and copied them to {args.paper_out}")


if __name__ == "__main__":
    main()
