"""Figures summarising the results (matplotlib only).

Usage: python scripts/08_figures.py [fig1 fig2 ...]   (default: all)
Figures are written as 600-dpi PNG and vector PDF to outputs/figures/.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "savefig.dpi": 600, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False,
})
# Okabe-Ito color-blind safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7",
      "grey": "#8C8C8C"}
LABEL = {"persistence": "Persistence", "linear_trend": "Theil–Sen trend",
         "damped_trend": "Damped trend", "gradient_boosting": "Gradient boosting",
         "gru": "GRU", "direct": "Direct MLP", "trend_residual": "Trend-residual",
         "gated": "Reliability-gated TR"}
COLOR = {"persistence": OI["grey"], "linear_trend": OI["sky"], "damped_trend": OI["yellow"],
         "gradient_boosting": OI["orange"], "gru": OI["purple"], "direct": OI["green"],
         "trend_residual": OI["blue"], "gated": OI["red"]}
H = len(C.HORIZONS)


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


def save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(C.FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#333333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color=color, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ Figure 1
def fig1():
    fig = plt.figure(figsize=(5.4, 4.3))
    # (a) rolling-origin design
    ax = fig.add_axes([0.02, 0.60, 0.96, 0.36])
    ax.set_xlim(-1.2, 15.6)
    ax.set_ylim(-0.3, 3.3)
    ax.axis("off")
    panel(ax, "a", x=0.0, y=0.98)
    ax.annotate("", xy=(9.0, 0.6), xytext=(-0.6, 0.6),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333"))
    ax.text(9.0, 0.25, "time (days)", ha="right", fontsize=7)
    rng = np.random.default_rng(3)
    times = np.sort(np.concatenate([np.arange(0, 8.6, 1.0), np.arange(0, 8.6, 1.0) + 0.25,
                                    rng.uniform(0, 8.6, 6)]))
    ax.plot(times, np.full_like(times, 0.6), "|", color="#555555", ms=7, mew=0.8)
    for k, (o, y0) in enumerate(((3.0, 1.35), (4.0, 2.05), (5.0, 2.75))):
        ax.add_patch(plt.Rectangle((o - 2, y0 - 0.22), 2, 0.44, fc="#DCEBF7", ec=OI["blue"], lw=0.7))
        ax.text(o - 1, y0, "48 h history", ha="center", va="center", fontsize=6.5)
        ax.plot([o], [y0], marker="v", color=OI["red"], ms=6)
        for j, (h, c) in enumerate(((1, OI["green"]), (2, OI["orange"]), (3, OI["purple"]))):
            ax.plot([o + h], [y0], "o", color=c, ms=4.5)
        ax.plot([o, o + 3], [y0, y0], ":", color="#777777", lw=0.7, zorder=0)
        ax.plot([o, o], [0.6, y0 - 0.22], "-", color=OI["red"], lw=0.5, alpha=0.6)
    ax.text(9.4, 2.75, "origin $t_0$ (▼), targets $t_0$ + 24/48/72 h (●)", fontsize=6.5,
            va="center")
    ax.text(9.4, 2.05, "next origin ≥ 24 h later (rolling origins)", fontsize=6.5, va="center")
    ax.text(9.4, 1.35, "first origin of each plant = early origin", fontsize=6.5,
            va="center")
    ax.text(-0.6, 1.0, "frames", fontsize=6.5, color="#555555")

    # (b) model
    ax = fig.add_axes([0.02, 0.0, 0.96, 0.56])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(ax, "b", x=0.0, y=0.97)
    box(ax, 0.01, 0.62, 0.15, 0.24, "RGB frames\nin history\nwindow", "#F2F2F2")
    box(ax, 0.21, 0.73, 0.23, 0.17, "Protocol A:\nannotation-derived N, a", "#FFF4D6", fs=6.5)
    box(ax, 0.21, 0.50, 0.23, 0.19, "Protocol B: ResNet-18\nmeasurement, cross-fitted\n(+ TTA spread)",
        "#E3F2E9", fs=6.5)
    arrow(ax, 0.16, 0.77, 0.21, 0.81)
    arrow(ax, 0.16, 0.70, 0.21, 0.60)
    box(ax, 0.48, 0.60, 0.17, 0.20, "History of\nN(t), a(t)\n(t ≤ $t_0$)", "#F2F2F2")
    arrow(ax, 0.44, 0.81, 0.48, 0.74)
    arrow(ax, 0.44, 0.60, 0.48, 0.66)
    box(ax, 0.71, 0.80, 0.27, 0.14, "Theil–Sen trend  T($t_0$+h)", "#DCEBF7")
    box(ax, 0.71, 0.57, 0.27, 0.16, "Residual network  s·f(x)\nx: history descriptors\n(optional image embedding)",
        "#DCEBF7")
    box(ax, 0.71, 0.33, 0.27, 0.17, "Reliability gate  g(r) $\\in$ (0, 1)\nr: TTA spread, fit residual,\nframe count, span",
        "#FBE3D6")
    arrow(ax, 0.65, 0.74, 0.71, 0.86)
    arrow(ax, 0.65, 0.68, 0.71, 0.65)
    arrow(ax, 0.65, 0.63, 0.71, 0.44)
    box(ax, 0.40, 0.07, 0.58, 0.15,
        "Forecast   ŷ($t_0$+h) = T($t_0$+h) + g(r) · s · f(x),   h = 24, 48, 72 h\n"
        "g ≡ 1: trend-residual;  g ≡ 0: Theil–Sen extrapolation", "#FFFFFF", fs=7.5)
    arrow(ax, 0.845, 0.57, 0.845, 0.50)
    arrow(ax, 0.845, 0.33, 0.845, 0.22)
    ax.plot([0.98, 0.995, 0.995], [0.87, 0.87, 0.145], color="#333333", lw=0.8)
    arrow(ax, 0.995, 0.145, 0.98, 0.145)
    ax.text(0.86, 0.525, "× g", fontsize=7, va="center")
    box(ax, 0.01, 0.07, 0.33, 0.30, "Evaluation\n• plant-level split (190/47/37/44)\n• 5 seeds, plant bootstrap CIs\n"
        "• split-conformal intervals\n  on calibration plants", "#F7F7F7", fs=6.8)
    save(fig, "fig1_design")


# ------------------------------------------------------------------ Figure 2
def fig2():
    meas = pd.read_csv(C.MEAS_DIR / "measurements.csv")
    meas["t"] = pd.to_datetime(meas.time)
    first = meas.groupby("plant").t.transform("min")
    meas["day"] = (meas.t - first).dt.total_seconds() / 86400
    fig = plt.figure(figsize=(5.4, 3.9))
    # (a) example plant
    plant = "Col-0|rep_21"
    d = meas[meas.plant == plant].sort_values("t")
    picks = np.linspace(0, len(d) - 1, 6).round().astype(int)
    for k, i in enumerate(picks):
        ax = fig.add_axes([0.02 + k * 0.163, 0.66, 0.15, 0.30])
        r = d.iloc[i]
        g, rep = r.plant.split("|")
        im = Image.open(C.RAW_DIR / g / rep / r.filename).convert("RGB")
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"day {r.day:.1f}\nN = {int(r.N)},  a = {r.a * 100:.1f}%", fontsize=6.5, pad=2)
        if k == 0:
            panel(ax, "a", x=-0.02, y=1.22)
    # (b,c) trajectories
    for k, (col, lab, scale) in enumerate((("N", "Visible leaf count N", 1),
                                           ("a", "Projected area a (% of image)", 100))):
        ax = fig.add_axes([0.08 + k * 0.48, 0.08, 0.40, 0.46])
        for p, g in meas.groupby("plant"):
            ax.plot(g.day, g[col] * scale, color="#9DB9D3", lw=0.3, alpha=0.35)
        bins = np.arange(0, 38, 1)
        meas["dbin"] = np.digitize(meas.day, bins)
        q = meas.groupby("dbin")[col].quantile([0.25, 0.5, 0.75]).unstack() * scale
        n = meas.groupby("dbin").size()
        q = q[n >= 30]
        x = bins[q.index - 1] + 0.5
        ax.fill_between(x, q[0.25], q[0.75], color=OI["blue"], alpha=0.25, lw=0)
        ax.plot(x, q[0.5], color=OI["blue"], lw=1.4, label="median (IQR)")
        ax.set_xlabel("Days since first image")
        ax.set_ylabel(lab)
        ax.legend(frameon=False, loc="upper left")
        panel(ax, "bc"[k], x=-0.17, y=1.02)
    save(fig, "fig2_data")


# ------------------------------------------------------------------ Figure 3
def fig3():
    meas = pd.read_csv(C.MEAS_DIR / "measurements.csv")
    acc = json.load(open(C.MEAS_DIR / "accuracy.json"))
    te = meas[meas.split == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0), gridspec_kw=dict(wspace=0.55))
    ax = axes[0]
    lim = (0, 28)
    ax.hexbin(te.N, te.N_meas, gridsize=27, extent=lim + lim, cmap="Blues", mincnt=1,
              linewidths=0, bins="log")
    ax.plot(lim, lim, "--", color=OI["red"], lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Annotated leaf count"); ax.set_ylabel("Measured leaf count")
    r = acc["cross_fitted"]["test"]
    ax.text(0.04, 0.96, f"MAE = {r['N_mae']:.2f}\nr = {r['N_r']:.3f}", transform=ax.transAxes,
            va="top", fontsize=6.5)
    panel(ax, "a")
    ax = axes[1]
    lim = (0, 45)
    ax.hexbin(te.a * 100, te.a_meas * 100, gridsize=27, extent=lim + lim, cmap="Greens",
              mincnt=1, linewidths=0, bins="log")
    ax.plot(lim, lim, "--", color=OI["red"], lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Annotated area (%)"); ax.set_ylabel("Measured area (%)")
    ax.text(0.04, 0.96, f"MAE = {r['a_mae'] * 100:.2f} pp\nr = {r['a_r']:.3f}",
            transform=ax.transAxes, va="top", fontsize=6.5)
    panel(ax, "b")
    ax = axes[2]
    splits = ["train", "val", "calib", "test"]
    x = np.arange(len(splits))
    ins = [acc["in_sample_full_model"][s]["N_mae"] for s in splits]
    cf = [acc["cross_fitted"][s]["N_mae"] for s in splits]
    ax.bar(x - 0.19, ins, 0.36, color=OI["grey"], label="model trained on all\ntraining plants")
    ax.bar(x + 0.19, cf, 0.36, color=OI["blue"], label="cross-fitted\n(training plants)")
    ax.set_xticks(x)
    ax.set_xticklabels(["train", "val", "calib", "test"])
    ax.set_ylabel("Leaf-count MAE (leaves)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.12))
    panel(ax, "c")
    save(fig, "fig3_measurement")


MAIN = ["persistence", "linear_trend", "damped_trend", "gradient_boosting", "gru", "direct",
        "trend_residual", "gated"]


def fig4():
    m = pd.read_csv(C.RESULT_DIR / "metrics.csv")
    m = m[(m.sigma == 0) & (m.subset == "test")]
    fig, axes = plt.subplots(2, 3, figsize=(5.4, 4.4),
                             gridspec_kw=dict(hspace=0.30, wspace=0.34, width_ratios=[1, 1, 0.85]))
    off = np.linspace(-0.28, 0.28, len(MAIN))
    for c, (src, title) in enumerate((("manual", "Protocol A"),
                                      ("measured", "Protocol B"))):
        for r, (tgt, ylab) in enumerate((("N", "Leaf-count MAE (leaves)"),
                                         ("a", "Area MAE (percentage points)"))):
            ax = axes[r, c]
            for k, name in enumerate(MAIN):
                row = m[(m.source == src) & (m.model == name)].iloc[0]
                est = [row[f"{tgt}_{h}"] for h in C.HORIZONS]
                lo = [row[f"{tgt}_{h}_lo"] for h in C.HORIZONS]
                hi = [row[f"{tgt}_{h}_hi"] for h in C.HORIZONS]
                x = np.arange(H) + off[k]
                ax.errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o-",
                            ms=3, lw=0.9, elinewidth=0.6, capsize=1.5, color=COLOR[name],
                            label=LABEL[name])
            ax.set_xticks(range(H))
            ax.set_xticklabels([f"{h} h" for h in C.HORIZONS])
            ax.set_ylabel(ylab)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(title, pad=12)
            if r == 1:
                ax.set_xlabel("Forecast horizon")
            panel(ax, "abcdef"[r * 3 + c], x=-0.3, y=1.06)
    stage_panels(axes[:, 2])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=6)
    save(fig, "fig4_accuracy")


def stage_panels(axes):
    """Forecast error by developmental stage (days since first image)."""
    from leafcast.stats import cluster_bootstrap_mean
    bins = [0, 5, 10, 15, 20, 40]
    labels = ["0–5", "5–10", "10–15", "15–20", ">20"]
    for src, col, lab in (("manual", OI["blue"], "GRU, Protocol A"),
                          ("measured", OI["red"], "GRU, Protocol B")):
        meta = pd.read_csv(C.PRED_DIR / src / "samples.csv")
        y = np.load(C.PRED_DIR / src / "targets.npy")
        te = np.where(meta.split == "test")[0]
        P = np.stack([np.load(f)["pred"] for f in sorted((C.PRED_DIR / src).glob("gru_s*.npz"))])
        E = np.abs(P[:, te] - y[te]).mean(0)
        stage = np.digitize(meta.age.values[te], bins[1:-1])
        for r, (sl, scale) in enumerate(((slice(0, H), 1), (slice(H, 2 * H), 100))):
            est, lo, hi = [], [], []
            for k in range(len(labels)):
                sel = stage == k
                e, l, h_, _ = cluster_bootstrap_mean(E[sel, sl].mean(1) * scale,
                                                     meta.plant.values[te][sel], n_boot=1000)
                est.append(e); lo.append(l); hi.append(h_)
            x = np.arange(len(labels)) + (-0.1 if src == "manual" else 0.1)
            axes[r].errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o-",
                             ms=3, lw=0.9, capsize=1.5, elinewidth=0.6, color=col, label=lab)
    for r, ax in enumerate(axes):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5.5, rotation=35)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Leaf-count MAE (leaves)" if r == 0 else "Area MAE (pp)")
        panel(ax, "cf"[r], x=-0.3, y=1.06)
        if r == 0:
            ax.set_title("GRU error by stage", pad=12)
            ax.legend(frameon=False, fontsize=6, loc="lower right")
        else:
            ax.set_xlabel("Days monitored at origin")


def fig5():
    comp = pd.read_csv(C.RESULT_DIR / "comparisons.csv")
    comp = comp[(comp.subset == "test") & comp.family.str.contains("visual_increment")]
    order = [("A", "direct+img_task vs direct", "Direct MLP + task emb."),
             ("A", "direct+img_imagenet vs direct", "Direct MLP + ImageNet emb."),
             ("A", "trend_residual+img_task vs trend_residual", "TR + task emb."),
             ("A", "trend_residual+img_imagenet vs trend_residual", "TR + ImageNet emb."),
             ("B", "direct+img_task vs direct", "Direct MLP + task emb."),
             ("B", "direct+img_imagenet vs direct", "Direct MLP + ImageNet emb."),
             ("B", "trend_residual+img_task vs trend_residual", "TR + task emb."),
             ("B", "trend_residual+img_imagenet vs trend_residual", "TR + ImageNet emb."),
             ("B", "gated+img_task vs gated", "Gated TR + task emb.")]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    ypos = []
    y = 0
    for k, (p, _, _) in enumerate(order):
        if k and order[k - 1][0] != p:
            y += 0.8
        ypos.append(y)
        y += 1
    ypos = -np.array(ypos)
    for j, (tgt, xlab) in enumerate((("N", "Δ leaf-count MAE (leaves)"),
                                     ("a", "Δ area MAE (percentage points)"))):
        ax = axes[j]
        for k, (p, lab, _) in enumerate(order):  # noqa: B007
            r = comp[(comp.family == f"{p}_visual_increment") & (comp.comparison == lab) &
                     (comp.target == tgt)].iloc[0]
            col = OI["blue"] if p == "A" else OI["red"]
            ax.errorbar(r["diff"], ypos[k], xerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]],
                        fmt="o", ms=3.5, color=col, elinewidth=0.9, capsize=2)
            star = "*" if r.p_holm < 0.05 else ""
            ax.text(r.ci_high, ypos[k] + 0.18, f" {star}", fontsize=8, color=col, va="center")
        ax.axvline(0, color="#555555", lw=0.6, ls="--")
        ax.set_xlabel(xlab)
        ax.spines["left"].set_visible(j == 0)
        panel(ax, "ab"[j], x=-0.04 if j else -0.62, y=1.0)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([o[2] for o in order])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    for ax in axes:
        ax.set_ylim(ypos.min() - 0.6, ypos.max() + 1.0)
    axes[0].text(-0.62, ypos[0] + 0.62, "Protocol A",
                 transform=axes[0].get_yaxis_transform(), color=OI["blue"], fontsize=7,
                 fontweight="bold")
    axes[0].text(-0.62, ypos[4] + 0.62, "Protocol B",
                 transform=axes[0].get_yaxis_transform(), color=OI["red"], fontsize=7,
                 fontweight="bold")
    save(fig, "fig5_visual_increment")


def _gates(cfg, sigma):
    files = [f for f in sorted((C.PRED_DIR / "measured").glob(f"{cfg}_s*.npz"))
             if f.stem.rsplit("_s", 1)[0] == cfg]
    meta = pd.read_csv(C.PRED_DIR / "measured" / "samples.csv")
    te = np.where(meta.split == "test")[0]
    vals = []
    for f in files:
        z = np.load(f)
        if sigma == 0:
            vals.append(z["gate"][te].mean(0))
        else:
            vals.append(np.mean([z[k] for k in z.files if k.startswith(f"gate_noise_{sigma}_")], 0).mean(0))
    return np.array(vals)                       # (seeds, 6)


def fig6():
    m = pd.read_csv(C.RESULT_DIR / "metrics.csv")
    m = m[(m.subset == "test")]
    sig = [0.0, 0.5, 1.0, 2.0, 3.0]
    fig, axes2 = plt.subplots(2, 2, figsize=(5.4, 4.4), gridspec_kw=dict(wspace=0.5, hspace=0.5))
    axes = axes2.ravel()
    models = ["persistence", "linear_trend", "gradient_boosting", "gru", "direct",
              "trend_residual", "gated"]
    for j, tgt in enumerate(("N", "a")):
        ax = axes[j]
        for name in models + ["trend_residual_noaug"]:
            src = m[(m.source == "measured") & (m.model == name)].set_index("sigma")
            v = [src.loc[s, f"{tgt}_mean"] for s in sig]
            base = name.replace("_noaug", "")
            ax.plot(sig, v, "o-" if "noaug" not in name else "s--", ms=2.8, lw=0.9,
                    color=COLOR[base], label=LABEL[base] + (" (no noise training)" if "noaug" in name else ""))
        ax.set_xlabel("Added noise σ (leaves)")
        ax.set_ylabel("Leaf-count MAE (leaves)" if tgt == "N" else "Area MAE (pp)")
        ax.set_xticks([0, 1, 2, 3])
        panel(ax, "ab"[j], x=-0.25)
    ax = axes[2]
    G = np.array([_gates("gated", s) for s in sig])           # (sigma, seeds, 6)
    for k, (h, c) in enumerate(zip(C.HORIZONS, (OI["green"], OI["orange"], OI["purple"]))):
        mu, sd = G[:, :, k].mean(1), G[:, :, k].std(1, ddof=1)
        ax.plot(sig, mu, "o-", ms=2.8, color=c, label=f"N, {h} h")
        ax.fill_between(sig, mu - sd, mu + sd, color=c, alpha=0.2, lw=0)
    ax.set_xlabel("Added noise σ (leaves)")
    ax.set_ylabel("Mean gate value (leaf count)")
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    panel(ax, "c", x=-0.25)
    ax = axes[3]
    comp = pd.read_csv(C.RESULT_DIR / "comparisons.csv")
    comp = comp[(comp.family == "leakage") & (comp.subset == "test") & (comp.target == "N")]
    names = ["direct_noaug", "trend_residual_noaug", "gru_noaug", "gated"]
    labs = ["Direct", "TR", "GRU", "Gated TR"]
    for k, n in enumerate(names):
        r = comp[comp.comparison == f"in-sample vs cross-fitted ({n})"].iloc[0]
        ax.bar(k, r["diff"], 0.6, color=OI["grey"])
        ax.errorbar(k, r["diff"], yerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]],
                    color="black", lw=0.7, capsize=2)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labs)
    ax.set_ylabel("Δ leaf-count MAE (leaves)\n(in-sample − cross-fitted)")
    panel(ax, "d", x=-0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1),
               fontsize=6)
    save(fig, "fig6_robustness")


def fig7():
    c = pd.read_csv(C.RESULT_DIR / "conformal.csv")
    cc = pd.read_csv(C.RESULT_DIR / "conformal_conditional.csv")
    sel = (c.source == "measured") & (c.model == "gru") & (c.subset == "test")
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0), gridspec_kw=dict(wspace=0.55))
    for j, (tgt, lab) in enumerate((("N", "leaves"), ("a", "pp"))):
        ax = axes[j]
        for meth, col, mk in (("standard", OI["grey"], "o"), ("normalized", OI["red"], "s")):
            for nom, ls in ((0.9, "-"), (0.95, "--")):
                d = c[sel & (c.target == tgt) & (c.method == meth) & np.isclose(c.nominal, nom)]
                g = d.groupby("horizon")[["coverage", "mean_width"]].mean()
                ax.plot(g.index, g.mean_width, mk + ls, ms=3, color=col, lw=0.9,
                        label=f"{meth}, {int(nom * 100)}%")
        ax.set_xticks(C.HORIZONS)
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel(f"Mean interval width ({lab})")
        ax.set_ylim(bottom=0)
        panel(ax, "ab"[j], x=-0.3)
    axes[0].legend(frameon=False, fontsize=6)
    ax = axes[2]
    d = cc[(cc.source == "measured") & (cc.model == "gru") & (cc.target == "N") &
           np.isclose(cc.nominal, 0.9)]
    g = d.groupby(["method", "noise_tertile"]).coverage.mean().unstack(0)
    x = np.arange(3)
    ax.bar(x - 0.19, g["standard"], 0.36, color=OI["grey"], label="standard")
    ax.bar(x + 0.19, g["normalized"], 0.36, color=OI["red"], label="normalized")
    ax.axhline(0.9, color="black", lw=0.7, ls="--")
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_xlabel("History noisiness tertile")
    ax.set_ylabel("Coverage (leaf count, 90%)")
    ax.legend(frameon=False, fontsize=6, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    panel(ax, "c", x=-0.3)
    save(fig, "fig7_intervals")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6,
        "fig7": fig7}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
