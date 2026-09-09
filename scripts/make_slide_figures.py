"""Generate every figure the 6-slide deck references, into deliverables/figures/.

Also copies the two plots already produced by notebooks/5-Transfer.ipynb so the
deck and the notebook stay consistent.

Palette matches the research deck (petrol blue / sienna / teal on warm paper).

    python scripts/make_slide_figures.py

Outputs (PNG, 300 dpi, transparent-safe):
  nb5_curve.png            slide 5   <- notebooks/5-Transfer.ipynb  cell 13
  nb5_confusion.png        (backup)  <- notebooks/5-Transfer.ipynb  cell 15
  slide2_pipeline.png      slide 2   pipeline: Madrid -> align -> predict
  slide2_scatter.png       slide 2   PCA of the two cities: raw vs aligned
  slide3_gains.png         slide 3   ascending contribution of each step
  slide4_perclass.png      slide 4   macro-F1 per age class
  slide5_curve.png         slide 5   F1 vs labels/class, restyled to the deck
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ---- palette ----
PAPER = "#FBFBF8"; INK = "#1A1C1F"; MUT = "#5B6470"
ACC = "#1F5C7A"; MADRID = "#A85D28"; AMS = "#2E7C8C"; RULE = "#D9DCE0"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": MUT, "font.size": 12, "svg.fonttype": "none",
})
for fam in ("IBM Plex Sans", "DejaVu Sans"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  wrote", name)


# ---- 0 · lift the two figures already in notebook 5 ----------------------
def extract_notebook_pngs():
    nb = json.loads((ROOT / "notebooks" / "5-Transfer.ipynb").read_text(encoding="utf-8"))
    got = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for o in cell.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if png:
                got.append(png)
    names = ["nb5_curve.png", "nb5_confusion.png"]
    for png, name in zip(got, names):
        (OUT / name).write_bytes(base64.b64decode(png))
        print("  wrote", name, "(from notebook 5)")


# ---- 1 · pipeline diagram (slide 2) -------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.1, "Madrid features\n(108 per pixel)", PAPER, INK),
        (3.5, "Line the cities up\ngroup by group  ×2", "#EAF1F4", ACC),
        (7.0, "Predict Amsterdam\n0.36 → 0.65", PAPER, AMS),
    ]
    w = 2.8
    centres = []
    for x, txt, fc, ec in boxes:
        b = FancyBboxPatch((x, 0.7), w, 1.6, boxstyle="round,pad=0.08,rounding_size=0.06",
                           linewidth=1.4, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w / 2, 1.5, txt, ha="center", va="center", fontsize=11.5, color=INK)
        centres.append(x + w)
    for x0, x1 in ((centres[0], boxes[1][0]), (centres[1], boxes[2][0])):
        ax.add_patch(FancyArrowPatch((x0 + .05, 1.5), (x1 - .05, 1.5),
                     arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color=MUT))
    ax.text(5, 0.25, "no Amsterdam labels used — only the model’s own guesses",
            ha="center", fontsize=9.5, color=MUT, style="italic")
    save(fig, "slide2_pipeline.png")


# ---- 2 · before / after PCA scatter (slide 2) --------------------------
def scatter():
    from sklearn.decomposition import PCA
    from src.data import build_city
    from src.adapt import make_coral, class_conditional_coral
    from sklearn.ensemble import RandomForestClassifier

    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    rng = np.random.default_rng(0)
    mi = rng.choice(len(m.X), 3000, replace=False)
    ai = rng.choice(len(a.X), 3000, replace=False)

    def panel(ax, Xm, Xa, title):
        p = PCA(n_components=2, random_state=0).fit(np.vstack([Xm, Xa]))
        pm, pa = p.transform(Xm), p.transform(Xa)
        ax.scatter(pm[:, 0], pm[:, 1], s=5, c=MADRID, alpha=.35, linewidths=0, label="Madrid")
        ax.scatter(pa[:, 0], pa[:, 1], s=5, c=AMS, alpha=.35, linewidths=0, label="Amsterdam")
        ax.set_title(title, fontsize=12, color=INK, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(RULE)

    rf = class_conditional_coral(
        m.X, m.y, a.X,
        lambda: RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                       random_state=0, n_jobs=-1), rounds=2)
    # the aligned Madrid used inside; reproduce final alignment for the plot
    coral = make_coral(m.X, a.X)
    Xm_pooled = coral(m.X)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    panel(axes[0], m.X[mi], a.X[ai], "Raw features — the cities sit apart")
    panel(axes[1], Xm_pooled[mi], a.X[ai], "After the line-up — they overlap")
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, markerscale=2)
    fig.suptitle("Fig.  Madrid vs Amsterdam in feature space",
                 fontsize=10, color=MUT, y=1.02)
    save(fig, "slide2_scatter.png")


# ---- 3 · ascending gains (slide 3) -----------------------------------
def gains():
    steps = [
        ("Baseline\n(provided)", 0.42, MUT),
        ("Line up\ncities", 0.49, ACC),
        ("Untangle\nfeatures", 0.62, ACC),
        ("Two models\nvote", 0.65, ACC),
        ("Neighbours\nvote", 0.66, ACC),
        ("500-setup\nsearch", 0.662, AMS),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    xs = np.arange(len(steps))
    vals = [v for _, v, _ in steps]
    cols = [c for _, _, c in steps]
    ax.bar(xs, vals, width=.62, color=cols, edgecolor="none")
    for x, v in zip(xs, vals):
        ax.text(x, v + .008, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([s for s, _, _ in steps], fontsize=9.5)
    ax.set_ylim(0.38, 0.72)
    ax.set_ylabel("macro-F1  (5 labels / class)")
    ax.axhline(0.664, color=MADRID, ls="--", lw=1.3)
    ax.text(0.05, 0.672, "Madrid in-city  0.66", ha="left", fontsize=9, color=MADRID)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Each step's contribution", fontsize=12, color=INK, pad=10)
    save(fig, "slide3_gains.png")


# ---- 4 · per-class F1 (slide 4) ------------------------------------
def perclass():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from src.data import build_city
    from src.adapt import class_conditional_coral, zca_whiten, spatial_smoother

    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    cls = np.unique(m.y)
    rf = class_conditional_coral(
        m.X, m.y, a.X,
        lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1), rounds=2)
    prior = rf.predict_proba(a.X)
    sm = spatial_smoother(a.pixels.to_numpy(float), k=8)
    n = 100
    Z = zca_whiten(a.X, shrink=0.2)(a.X)
    rng = np.random.default_rng(42)
    runs = []
    for _ in range(15):                       # average per-class F1 over draws
        sup = np.concatenate([rng.choice(np.where(a.y == c)[0], n, replace=False) for c in cls])
        q = np.setdiff1d(np.arange(len(a.y)), sup)
        head = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=42, n_jobs=-1).fit(Z[sup], a.y[sup])
        field = 0.9 * head.predict_proba(Z) + 0.1 * prior
        pred = cls[sm(field)[q].argmax(1)]
        runs.append(f1_score(a.y[q], pred, average=None, labels=cls))
    per = np.mean(runs, axis=0)
    err = np.std(runs, axis=0)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    names = ["1\npre-1945", "2\n1945-84", "3\n1984-04", "4\n2004-24"]
    lo = float(min(per))
    cols = [MADRID if abs(v - lo) < 1e-9 else AMS for v in per]
    bars = ax.bar(names, per, yerr=err, width=.6, color=cols,
                  error_kw=dict(ecolor=MUT, capsize=4, lw=1.2))
    for b, v in zip(bars, per):
        ax.text(b.get_x() + b.get_width() / 2, v + .04, f"{v:.2f}", ha="center", fontsize=11, color=INK)
    ax.set_ylim(0, .95); ax.set_ylabel("F1  (100 labels/class, 15 draws)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Class 1 (pre-1945) is the strongest; the others cluster\n"
                 "near 0.7 — no single unsolvable pair",
                 fontsize=11.5, color=INK, pad=10)
    save(fig, "slide4_perclass.png")


# ---- 5 · restyled F1 curve (slide 5) -----------------------------
def curve():
    import csv
    rows = list(csv.reader((ROOT / "results" / "deliverable_table.csv").open()))
    d = {r[0]: (float(r[1]), float(r[2]) if r[2] else 0.0) for r in rows[1:]}
    shots = [5, 25, 50, 100, 200]
    mean = [d[f"Amsterdam few-shot, {s}/class"][0] for s in shots]
    sd = [d[f"Amsterdam few-shot, {s}/class"][1] for s in shots]
    mad = d["Madrid 5x5 CV"][0]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.errorbar(shots, mean, yerr=sd, fmt="o-", color=AMS, lw=2.4, capsize=4,
                markersize=6, label="Madrid → Amsterdam")
    ax.axhline(mad, color=MADRID, ls="--", lw=1.4, label=f"Madrid in-city {mad:.2f}")
    ax.axhline(d["Amsterdam zero-shot (class-cond CORAL)"][0], color=ACC, ls=":", lw=1.4,
               label=f"zero labels {d['Amsterdam zero-shot (class-cond CORAL)'][0]:.2f}")
    ax.set_xscale("log", base=2); ax.set_xticks(shots)
    ax.set_xticklabels(shots)
    ax.set_xlabel("labelled pixels per class")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.60, 0.78)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[2], h[0], h[1]], [l[2], l[0], l[1]],
              frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Amsterdam transfer — macro-F1 vs label budget",
                 fontsize=11.5, color=INK, pad=10)
    save(fig, "fig_slide3_curve.png")




# ---- 6 · F1 score table (slide 3) --------------------------------
def score_table():
    import csv
    rows = list(csv.reader((ROOT / "results" / "deliverable_table.csv").open()))[1:]
    d = {r[0]: (float(r[1]), float(r[2]) if r[2] else None) for r in rows}

    def line(label, key, bold=False):
        m, s = d[key]
        val = f"{m:.2f}" + (f"  ± {s:.3f}" if s is not None else "")
        return label, val, bold

    data = [
        line("Madrid, tested on itself", "Madrid 5x5 CV"),
        ("", "", False),
        line("Amsterdam — zero labels", "Amsterdam zero-shot (raw)"),
        line("Amsterdam — zero labels, lined up", "Amsterdam zero-shot (class-cond CORAL)", True),
        ("", "", False),
        line("Amsterdam — 5 labels / class", "Amsterdam few-shot, 5/class"),
        line("Amsterdam — 25 labels / class", "Amsterdam few-shot, 25/class"),
        line("Amsterdam — 50 labels / class", "Amsterdam few-shot, 50/class", True),
        line("Amsterdam — 100 labels / class", "Amsterdam few-shot, 100/class"),
        line("Amsterdam — 200 labels / class", "Amsterdam few-shot, 200/class"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(data)
    top = 0.92
    ax.text(0.02, 0.985, "Setting", fontsize=11, color=MUT, weight="bold")
    ax.text(0.98, 0.985, "macro-F1", fontsize=11, color=MUT, weight="bold", ha="right")
    ax.plot([0.02, 0.98], [0.955, 0.955], color=INK, lw=1.8)
    for i, (label, val, bold) in enumerate(data):
        y = top - i * (top / n)
        if not label:
            ax.plot([0.02, 0.98], [y + 0.015, y + 0.015], color=RULE, lw=0.8)
            continue
        w = "bold" if bold else "normal"
        col = AMS if bold else INK
        ax.text(0.02, y, label, fontsize=11, color=INK, weight=w, va="center")
        ax.text(0.98, y, val, fontsize=11, color=col, weight=w, ha="right", va="center",
                family="DejaVu Sans Mono")
    ax.plot([0.02, 0.98], [top - n * (top / n) + 0.02, top - n * (top / n) + 0.02],
            color=INK, lw=1.8)
    ax.text(0.02, -0.03, "± = 1 SD over folds / 20 resamples · baseline (provided notebook): 0.42 → 0.67",
            fontsize=8.5, color=MUT)
    save(fig, "fig_slide3_table.png")


# ---- 7 · what didn't work (slide 4) --------------------------------
def didnt_work():
    rows = [
        ("Neural-net embedding",
         "the features were already separable — nothing left to learn"),
        ("Ordinal training",
         "made the mistakes smaller, not fewer; the score only counts right vs wrong"),
        ("Self-training on its own guesses",
         "it learns its own early mistakes and compounds them"),
        ("Gradient boosting",
         "matches us with plenty of labels, then collapses at 5"),
        ("Correcting the age mix",
         "needs confidence scores that the city gap distorts"),
    ]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.02, 0.95, "The clever fixes we dropped",
            fontsize=15, color=INK, weight="bold")
    ax.plot([0.02, 0.98], [0.895, 0.895], color=INK, lw=1.8)

    top, step = 0.80, 0.150
    for i, (what, why) in enumerate(rows):
        y = top - i * step
        ax.text(0.02, y + 0.006, "✗", fontsize=16, color=MADRID,
                va="center", weight="bold")
        ax.text(0.075, y + 0.030, what, fontsize=13.5, color=INK,
                va="center", weight="bold")
        ax.text(0.075, y - 0.034, why, fontsize=11.5, color=MUT, va="center")
        if i < n - 1:
            ax.plot([0.075, 0.98], [y - 0.075, y - 0.075], color=RULE, lw=0.8)

    base = top - (n - 1) * step - 0.075
    ax.plot([0.02, 0.98], [base, base], color=INK, lw=1.8)
    ax.text(0.02, base - 0.09,
            "Every attempt to out-model the data lost to using it more carefully.",
            fontsize=12, color=ACC, style="italic", weight="bold")
    save(fig, "fig_slide4_didntwork.png")

if __name__ == "__main__":
    import time
    t = time.time()
    print("figures ->", OUT)
    extract_notebook_pngs()
    pipeline()
    gains()
    curve()
    perclass()
    scatter()
    scatter_single()
    scatter_after()
    scatter_before_after()
    score_table()
    didnt_work()
    spatial_audit()
    print(f"done in {time.time()-t:.0f}s")


# ---- 8 · spatial-block audit (slide 5) --------------------------------
def spatial_audit():
    import csv
    import json
    budgets = [25, 50, 100, 200]
    zs = 0.6513                                   # zero-shot class-cond CORAL, whole city
    tbl = {r[0]: float(r[1]) for r in
           csv.reader((ROOT / "results" / "deliverable_table.csv").open())
           if r and r[0].startswith("Amsterdam few-shot")}
    rnd = [tbl[f"Amsterdam few-shot, {n}/class"] - zs for n in budgets]
    g2 = json.loads((ROOT / "results" / "spatial_block_eval_gap2.json").read_text())["per_budget"]
    spat = [g2[str(n)]["vs_zero_shot"] for n in budgets]
    se = [g2[str(n)]["std"] / (g2[str(n)]["n_folds"] ** 0.5) for n in budgets]

    x = np.arange(len(budgets))
    nl = chr(10)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axhline(0, color=MUT, lw=1, ls=(0, (4, 3)))
    ax.text(x[-1] + 0.06, 0, "  zero-shot" + nl + "  (no local labels)", va="center",
            fontsize=10, color=MUT)

    ax.plot(x, rnd, "o-", color=ACC, lw=2.4, ms=7, label="random split  (our deliverable)")
    ax.plot(x, spat, "s-", color=MADRID, lw=2.4, ms=6,
            label="labels held a map-tile away")
    ax.fill_between(x, np.array(spat) - np.array(se), np.array(spat) + np.array(se),
                    color=MADRID, alpha=0.15, lw=0)

    for xi, yv in zip(x, rnd):
        ax.annotate(f"+{yv:.2f}", (xi, yv), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color=ACC, weight="bold")
    ax.text(1.5, 0.030, "the gap between the lines" + nl + "is proximity, not the labels",
            fontsize=10.5, color=INK, ha="center", style="italic")

    ax.set_xticks(x); ax.set_xticklabels(budgets)
    ax.set_xlabel("labelled Amsterdam pixels per class")
    ax.set_ylabel("macro-F1 gained from the local labels")
    ax.set_xlim(-0.3, len(budgets) - 0.15 + 0.9)
    ax.set_title("How much of the few-shot gain is real?", fontsize=13, color=INK, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0, -0.235,
            "Same pipeline, same budgets. Support pixels drawn only from map tiles a full "
            "tile from" + nl + "the scored area, so none sit next to a scored pixel. "
            "Per-tile macro-F1 is noisy (band = SE).",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color=MUT)
    save(fig, "fig_slide5_spatial.png")

# ---- 1b · single-panel domain-shift scatter (slide 1) ---------------

def scatter_single():
    """Overwrite fig_slide1_scatter.png with a single, standardized panel that
    actually shows the domain shift: z-scored features, PCA to 2-D, per-city
    density contours + centroids so the offset reads even where dots overlap."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
    from src.data import build_city

    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    rng = np.random.default_rng(0)
    mi = rng.choice(len(m.X), 3500, replace=False)
    ai = rng.choice(len(a.X), 3500, replace=False)

    sc = StandardScaler().fit(np.vstack([m.X[mi], a.X[ai]]))
    p = PCA(n_components=2, random_state=0).fit(sc.transform(np.vstack([m.X[mi], a.X[ai]])))
    pm = p.transform(sc.transform(m.X[mi]))
    pa = p.transform(sc.transform(a.X[ai]))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    lo = np.percentile(np.vstack([pm, pa]), 1, axis=0)
    hi = np.percentile(np.vstack([pm, pa]), 99, axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    xx, yy = np.mgrid[lo[0]:hi[0]:120j, lo[1]:hi[1]:120j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for pts, col, lab in ((pm, MADRID, "Madrid"), (pa, AMS, "Amsterdam")):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=col, alpha=0.12, linewidths=0)
        d = gaussian_kde(pts.T)(grid).reshape(xx.shape)
        ax.contour(xx, yy, d, levels=4, colors=col, linewidths=1.4, alpha=0.9)
        c = pts.mean(axis=0)
        ax.plot(c[0], c[1], "X", ms=15, mfc=col, mec="white", mew=2, label=lab)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=11, handletextpad=0.3)
    for t, col in zip(leg.get_texts(), (MADRID, AMS)):
        t.set_color(col); t.set_fontweight("bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(RULE)
    ax.set_title("Same features, two cities — the distributions are offset",
                 fontsize=12.5, color=INK, pad=12)
    ax.text(0.5, -0.05,
            "108 standardized features → 2-D · contours = density, X = centre of each city",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=MUT)
    save(fig, "fig_slide1_scatter.png")


# ---- 2b · before / after the line-up (slide 2) ----------------------
def scatter_before_after():
    """fig_slide2_scatter.png - two panels sharing one projection so Madrid's
    cloud is seen to *move* onto Amsterdam after the line-up. z-scored 108
    features -> PCA (fit on the raw combined data, reused for both panels);
    per-city density contours + centroid X, matching the slide-1 style."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
    from src.data import build_city
    from src.adapt import make_coral

    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    rng = np.random.default_rng(0)
    mi = rng.choice(len(m.X), 3500, replace=False)
    ai = rng.choice(len(a.X), 3500, replace=False)
    Xm, Xa = m.X[mi], a.X[ai]

    coral = make_coral(m.X, a.X)          # fitted on the two unlabelled clouds
    Xm_aligned = coral(Xm)

    sc = StandardScaler().fit(np.vstack([Xm, Xa]))
    pca = PCA(n_components=2, random_state=0).fit(sc.transform(np.vstack([Xm, Xa])))
    proj = lambda X: pca.transform(sc.transform(X))
    pm_raw, pa, pm_al = proj(Xm), proj(Xa), proj(Xm_aligned)

    allpts = np.vstack([pm_raw, pa, pm_al])
    lo = np.percentile(allpts, 1, axis=0)
    hi = np.percentile(allpts, 99, axis=0)
    xx, yy = np.mgrid[lo[0]:hi[0]:110j, lo[1]:hi[1]:110j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    def panel(ax, pm, title):
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        for pts, col in ((pm, MADRID), (pa, AMS)):
            ax.scatter(pts[:, 0], pts[:, 1], s=5, c=col, alpha=0.10, linewidths=0)
            d = gaussian_kde(pts.T)(grid).reshape(xx.shape)
            ax.contour(xx, yy, d, levels=4, colors=col, linewidths=1.3, alpha=0.9)
            c = pts.mean(axis=0)
            ax.plot(c[0], c[1], "X", ms=13, mfc=col, mec="white", mew=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(RULE)
        ax.set_title(title, fontsize=12, color=INK, pad=8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    panel(axes[0], pm_raw, "Before  ·  the cities sit apart")
    panel(axes[1], pm_al, "After the line-up  ·  they overlap")
    axes[1].plot([], [], "X", ms=11, mfc=MADRID, mec="white", mew=1.6, label="Madrid")
    axes[1].plot([], [], "X", ms=11, mfc=AMS, mec="white", mew=1.6, label="Amsterdam")
    leg = axes[1].legend(loc="upper right", frameon=False, fontsize=10, handletextpad=0.3)
    for t, col in zip(leg.get_texts(), (MADRID, AMS)):
        t.set_color(col); t.set_fontweight("bold")
    fig.text(0.5, -0.01,
             "108 standardized features → 2-D, one shared projection · "
             "contours = density, X = centre of each city",
             ha="center", fontsize=8.8, color=MUT)
    save(fig, "alt_slide2_scatter_2panel.png")


# ---- 2c · single "after the line-up" panel (slide 2, less clutter) ---
def scatter_after():
    """fig_slide2_scatter.png as ONE panel: the clouds after the line-up.
    Slide 1 already showed them apart; this is just the payoff. Same shared
    projection as scatter_before_after so the two stay consistent if swapped."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
    from src.data import build_city
    from src.adapt import make_coral

    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    rng = np.random.default_rng(0)
    Xm, Xa = m.X[rng.choice(len(m.X), 3500, replace=False)], a.X[rng.choice(len(a.X), 3500, replace=False)]
    Xm_al = make_coral(m.X, a.X)(Xm)

    sc = StandardScaler().fit(np.vstack([Xm, Xa]))
    pca = PCA(n_components=2, random_state=0).fit(sc.transform(np.vstack([Xm, Xa])))
    proj = lambda X: pca.transform(sc.transform(X))
    pa, pm = proj(Xa), proj(Xm_al)

    allp = np.vstack([pa, pm])
    lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    xx, yy = np.mgrid[lo[0]:hi[0]:120j, lo[1]:hi[1]:120j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    for pts, col, lab in ((pm, MADRID, "Madrid (lined up)"), (pa, AMS, "Amsterdam")):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=col, alpha=0.11, linewidths=0)
        d = gaussian_kde(pts.T)(grid).reshape(xx.shape)
        ax.contour(xx, yy, d, levels=4, colors=col, linewidths=1.4, alpha=0.9)
        c = pts.mean(axis=0)
        ax.plot(c[0], c[1], "X", ms=15, mfc=col, mec="white", mew=2, label=lab)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11, handletextpad=0.3)
    for t, col in zip(leg.get_texts(), (MADRID, AMS)):
        t.set_color(col); t.set_fontweight("bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(RULE)
    ax.set_title("After the line-up — Madrid now sits on Amsterdam",
                 fontsize=12.5, color=INK, pad=12)
    ax.text(0.5, -0.05,
            "same features as slide 1, where the two clouds sat apart · X = centre of each city",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=MUT)
    save(fig, "fig_slide2_scatter.png")
