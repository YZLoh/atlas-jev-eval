"""Figures for the experiment report.

Design notes
------------
The palette is a validated categorical set (slots blue / orange / aqua / yellow),
checked for colourblind separation and contrast rather than chosen by eye. Aqua
and yellow fall below 3:1 against the surface, so every mark that uses them
carries a **direct value label** - identity and magnitude are never conveyed by
colour alone. A legend appears whenever there are two or more series.

These are static PNGs for a repository and deliberately commit to a single light
look, with explicit colours rather than a theme-dependent default.

The most important behaviour here: **a figure with insufficient labelled data
says so on its face** rather than rendering an empty or misleading plot. A blank
axis reads as "zero"; a stated "insufficient data" reads as what it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from atlas_jev.config import FIGURES_DIR, PROCESSED_DIR

# -- validated palette ------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a83"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "figure.dpi": 130,
    }
)


def _style(ax: Any, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """Recessive chrome: no top/right spines, faint grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    # Only pass line properties when enabling; matplotlib warns (and silently
    # enables the grid) if properties accompany grid(False).
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    else:
        ax.grid(False, axis="y")
    if xgrid:
        ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    else:
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def _no_data(ax: Any, message: str) -> None:
    """Say plainly that there is not enough data, instead of drawing nothing."""
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=11,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _save(fig: Any, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _caption(fig: Any, text: str, offset: float = -0.04) -> None:
    fig.text(0.0, offset, text, fontsize=8, color=INK_MUTED, ha="left", va="top", wrap=True)


# ---------------------------------------------------------------------------
# Figure 1 - attack success rate by ATLAS technique
# ---------------------------------------------------------------------------
def figure_attack_success(summary: dict[str, Any]) -> Path:
    by_technique = summary.get("target_robustness_by_technique", {})
    rows = [
        (name, data["attack_success_rate"], data["attack_cases"])
        for name, data in by_technique.items()
        if data.get("attack_success_rate") is not None
    ]

    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.75 * max(1, len(rows)) + 1.2)))
    if not rows:
        _no_data(
            ax,
            "No labelled attack cases yet.\n"
            "Attack success rate requires human ground-truth labels\n"
            "(run: python -m atlas_jev label)",
        )
        ax.set_title("Figure 1  Attack success rate by ATLAS technique")
        return _save(fig, "fig1_attack_success_by_technique.png")

    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    counts = [r[2] for r in rows]

    positions = range(len(rows))
    ax.barh(list(positions), values, height=0.45, color=SERIES[0], zorder=3)
    for y, value, count in zip(positions, values, counts):
        # Direct label: magnitude never depends on reading the axis alone.
        ax.text(
            value + 0.015,
            y,
            f"{value:.0%}  (n={count})",
            va="center",
            fontsize=9,
            color=INK,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("attack success rate (ground truth)")
    ax.set_title("Figure 1  Attack success rate by ATLAS technique")
    _style(ax, xgrid=True, ygrid=False)
    _caption(
        fig,
        "Measured from human ground-truth labels, not from any evaluator. "
        "Control cases are excluded. n is the number of labelled attack cases.",
        offset=-0.14,
    )
    return _save(fig, "fig1_attack_success_by_technique.png")


# ---------------------------------------------------------------------------
# Figure 2 - evaluator precision / recall / F1
# ---------------------------------------------------------------------------
def figure_evaluator_reliability(summary: dict[str, Any]) -> Path:
    reliability = summary.get("evaluator_reliability", {})
    names = [n for n, c in reliability.items() if c.get("scored")]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if not names:
        _no_data(
            ax,
            "No evaluator predictions could be scored yet.\n"
            "Scoring needs both an evaluator decision and a ground-truth label.",
        )
        ax.set_title("Figure 2  Evaluator reliability")
        return _save(fig, "fig2_evaluator_reliability.png")

    metrics = ("precision", "recall", "f1")
    width = 0.26
    for index, metric in enumerate(metrics):
        offsets = [i + (index - 1) * (width + 0.012) for i in range(len(names))]
        values = [reliability[n][metric] or 0.0 for n in names]
        ax.bar(
            offsets,
            values,
            width=width,
            label=metric,
            color=SERIES[index],
            zorder=3,
        )
        for x, value, name in zip(offsets, values, names):
            raw = reliability[name][metric]
            # Direct labels are mandatory for the low-contrast slots, and
            # "n/a" states an undefined metric rather than drawing a zero bar
            # that would read as a real measurement of zero.
            ax.text(
                x,
                (value if raw is not None else 0) + 0.02,
                "n/a" if raw is None else f"{raw:.2f}",
                ha="center",
                fontsize=8,
                color=INK if raw is not None else INK_MUTED,
            )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{n}\n(scored {reliability[n]['scored']})" for n in names], fontsize=9
    )
    ax.set_ylim(0, 1.12)
    # Keep bar width constant regardless of how many evaluators ran, so a
    # single-evaluator run does not render as three enormous blocks.
    ax.set_xlim(-0.62, len(names) - 0.38)
    ax.set_ylabel("score")
    ax.set_title("Figure 2  Evaluator reliability vs human ground truth")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _style(ax)

    notes = []
    for name in names:
        confusion = reliability[name]
        if confusion.get("abstained"):
            notes.append(f"{name} abstained on {confusion['abstained']}")
        if confusion.get("unlabelled"):
            notes.append(f"{confusion['unlabelled']} record(s) unlabelled")
    _caption(
        fig,
        "Abstentions are excluded, never counted as negatives. "
        "'n/a' marks an undefined metric (e.g. precision with no positive "
        "predictions), which is not the same as zero. "
        + ("; ".join(dict.fromkeys(notes)) if notes else ""),
        offset=-0.20,
    )
    return _save(fig, "fig2_evaluator_reliability.png")


# ---------------------------------------------------------------------------
# Figure 3 - confidence vs correctness, and calibration
# ---------------------------------------------------------------------------
def figure_confidence_vs_correctness(summary: dict[str, Any]) -> Path:
    calibration = summary.get("calibration", {})
    candidates = [
        (name, data)
        for name, data in calibration.items()
        if data.get("reliability", {}).get("n")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    if not candidates:
        for ax in axes:
            _no_data(ax, "No labelled probability pairs yet.")
        fig.suptitle("Figure 3  Confidence vs correctness", fontsize=12, fontweight="bold")
        return _save(fig, "fig3_confidence_vs_correctness.png")

    # Left: reliability diagram - does p(yes) behave like a probability?
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=INK_MUTED,
            label="perfect calibration", zorder=2)
    for index, (name, data) in enumerate(candidates):
        bins = [b for b in data["reliability"]["bins"] if b["count"]]
        if not bins:
            continue
        xs = [b["mean_predicted"] for b in bins]
        ys = [b["observed_rate"] for b in bins]
        sizes = [40 + 22 * b["count"] for b in bins]
        colour = SERIES[index % len(SERIES)]
        # No connecting line: each bin is an independent estimate, and with
        # one or two samples per bin a line would imply a trend that is not
        # in the data. Marker area encodes bin count so sparse bins read as
        # sparse.
        ax.scatter(xs, ys, s=sizes, color=colour, edgecolor=SURFACE, linewidth=2,
                   zorder=4, label=name)
        for x, y, b in zip(xs, ys, bins):
            ax.text(x, y + 0.045, f"n={b['count']}", ha="center", fontsize=7,
                    color=INK_MUTED, zorder=5)
        ece = data["reliability"]["ece"]
        brier = data["reliability"]["brier"]
        ax.annotate(
            f"{name}\nECE {ece:.3f} · Brier {brier:.3f}",
            xy=(0.03, 0.78 - 0.14 * index),
            xycoords="axes fraction",
            fontsize=8,
            color=INK_SECONDARY,
            va="top",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title("Calibration (reliability diagram)", fontsize=11)
    ax.text(
        0.03,
        0.02,
        "bins with n=1 can only read 0 or 1,\nso they cannot sit on the diagonal",
        transform=ax.transAxes,
        fontsize=7,
        color=INK_MUTED,
        va="bottom",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, xgrid=True)

    # Right: does higher confidence mean more often correct?
    ax = axes[1]
    plotted = False
    for index, (name, data) in enumerate(candidates):
        cvc = data.get("confidence_vs_correctness", {})
        bins = [b for b in cvc.get("bins", []) if b["count"]]
        if not bins:
            continue
        plotted = True
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        colour = SERIES[index % len(SERIES)]
        ax.scatter(xs, ys, s=[40 + 22 * b["count"] for b in bins], color=colour,
                   edgecolor=SURFACE, linewidth=2, zorder=4, label=name)
        for x, y, b in zip(xs, ys, bins):
            ax.text(x, y + 0.035, f"n={b['count']}", ha="center", fontsize=7,
                    color=INK_MUTED)
        auc = cvc.get("auc_confidence_predicts_correct")
        source = ", ".join(cvc.get("confidence_sources", [])) or "unknown"
        ax.annotate(
            f"{name}: AUC {auc:.3f} (source: {source})" if auc is not None
            else f"{name}: AUC n/a",
            xy=(0.03, 0.10 + 0.08 * index),
            xycoords="axes fraction",
            fontsize=8,
            color=INK_SECONDARY,
        )
    if not plotted:
        _no_data(ax, "No answers carried both a confidence value\nand a ground-truth label.")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy within bin")
        ax.set_title("Does confidence predict correctness?", fontsize=11)
        # Only label series when there is more than one: a single-entry legend
        # sitting inside the axes is indistinguishable from a data point.
        if len(candidates) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        _style(ax, xgrid=True)

    fig.suptitle(
        "Figure 3  Confidence vs correctness", fontsize=12, fontweight="bold", y=1.02
    )
    _caption(
        fig,
        "Left: whether the probability is calibrated (H2a). Right: whether confidence "
        "predicts correctness (H2b); AUC near 0.5 means it carries no information. "
        "For Noul answers the 'confidence' is our derived 2|p-0.5|, not an API value; "
        "for Choice/Score it is the API's dispersion statistic. Marker area scales "
        "with bin count - small bins are noisy.",
    )
    return _save(fig, "fig3_confidence_vs_correctness.png")


# ---------------------------------------------------------------------------
# Figure 4 - detection performance across thresholds
# ---------------------------------------------------------------------------
def figure_threshold_sweep(summary: dict[str, Any]) -> Path:
    calibration = summary.get("calibration", {})
    # Only evaluators with a real probability to threshold belong here.
    candidates = [
        (name, data)
        for name, data in calibration.items()
        if any(p.get("scored") for p in data.get("threshold_sweep", []))
        and data.get("roc", {}).get("auc") is not None
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    if not candidates:
        for ax in axes:
            _no_data(ax, "Not enough labelled data for a threshold sweep.\n"
                         "Needs at least one positive and one negative label.")
        fig.suptitle("Figure 4  Detection performance vs confidence threshold",
                     fontsize=12, fontweight="bold")
        return _save(fig, "fig4_threshold_sweep.png")

    name, data = candidates[0]

    # Left: the four rates against threshold. Four series, each directly
    # labelled at its right-hand end as well as in the legend.
    ax = axes[0]
    sweep = [p for p in data["threshold_sweep"] if p.get("scored")]
    thresholds = [p["threshold"] for p in sweep]
    series = {
        "precision": [p["precision"] for p in sweep],
        "recall": [p["recall"] for p in sweep],
        "false positive rate": [p["false_positive_rate"] for p in sweep],
        "false negative rate": [p["false_negative_rate"] for p in sweep],
    }
    for index, (label, values) in enumerate(series.items()):
        xs = [t for t, v in zip(thresholds, values) if v is not None]
        ys = [v for v in values if v is not None]
        if not xs:
            continue
        colour = SERIES[index % len(SERIES)]
        ax.plot(xs, ys, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, color=colour,
                label=label, zorder=3)
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                    va="center")
    ax.set_xlabel("decision threshold on p(yes)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.04, 1.14)
    ax.set_xlim(min(thresholds) - 0.03, max(thresholds) + 0.22)
    ax.set_title(f"Threshold sweep — {name}", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    _style(ax, xgrid=True)

    # Right: ROC, which is threshold-independent.
    ax = axes[1]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=INK_MUTED,
            label="chance (AUC 0.5)", zorder=2)
    for index, (candidate_name, candidate) in enumerate(candidates):
        roc = candidate["roc"]
        points = sorted(roc["points"], key=lambda p: p["fpr"])
        ax.plot([p["fpr"] for p in points], [p["tpr"] for p in points],
                linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, color=SERIES[index % len(SERIES)],
                label=f"{candidate_name} (AUC {roc['auc']:.3f})", zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC (threshold-independent)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, xgrid=True)

    fig.suptitle("Figure 4  Detection performance vs confidence threshold",
                 fontsize=12, fontweight="bold", y=1.02)
    _caption(
        fig,
        "The whole sweep is recomputed from stored probabilities, so it costs no "
        "additional API calls. This addresses H2: whether the probability provides "
        "a usable control for the false-positive / false-negative trade-off.",
    )
    return _save(fig, "fig4_threshold_sweep.png")


# ---------------------------------------------------------------------------
# Figure 5 - latency
# ---------------------------------------------------------------------------
def figure_latency(summary: dict[str, Any]) -> Path:
    latency = summary.get("latency", {})
    rows: list[tuple[str, float, float, int]] = []
    target = latency.get("target", {})
    if target.get("count"):
        rows.append(("target model", target["mean_ms"], target["p95_ms"], target["count"]))
    for name, stats in (latency.get("evaluators") or {}).items():
        if stats.get("count"):
            rows.append((name, stats["mean_ms"], stats["p95_ms"], stats["count"]))

    fig, ax = plt.subplots(figsize=(7.5, max(2.4, 0.8 * max(1, len(rows)) + 1.2)))
    if not rows:
        _no_data(ax, "No latency measurements recorded.")
        ax.set_title("Figure 5  Latency")
        return _save(fig, "fig5_latency.png")

    rows.sort(key=lambda r: r[1])
    positions = list(range(len(rows)))
    height = 0.34
    ax.barh([p + height / 2 + 0.01 for p in positions], [r[1] for r in rows],
            height=height, color=SERIES[0], label="mean", zorder=3)
    ax.barh([p - height / 2 - 0.01 for p in positions], [r[2] for r in rows],
            height=height, color=SERIES[1], label="p95", zorder=3)

    widest = max(max(r[1], r[2]) for r in rows) or 1.0
    for position, row in zip(positions, rows):
        ax.text(row[1] + widest * 0.015, position + height / 2 + 0.01,
                f"{row[1]:.0f} ms", va="center", fontsize=8, color=INK)
        ax.text(row[2] + widest * 0.015, position - height / 2 - 0.01,
                f"{row[2]:.0f} ms", va="center", fontsize=8, color=INK)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{r[0]}\n(n={r[3]})" for r in rows], fontsize=9)
    ax.set_xlim(0, widest * 1.28)
    ax.set_xlabel("latency (ms)")
    ax.set_title("Figure 5  Latency: target model vs evaluators")
    ax.legend(frameon=False, ncols=2, loc="lower right")
    _style(ax, xgrid=True, ygrid=False)
    _caption(
        fig,
        "Wall-clock, including network time, measured on one machine - indicative, "
        "not a benchmark. Relevant to H3.",
    )
    return _save(fig, "fig5_latency.png")


# ---------------------------------------------------------------------------
def generate_all(summary_path: Path | None = None) -> list[Path]:
    """Render every figure from the processed summary."""
    summary_path = summary_path or PROCESSED_DIR / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"no summary at {summary_path}. Run `python -m atlas_jev evaluate` first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return [
        figure_attack_success(summary),
        figure_evaluator_reliability(summary),
        figure_confidence_vs_correctness(summary),
        figure_threshold_sweep(summary),
        figure_latency(summary),
    ]
