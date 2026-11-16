# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig5_cv_split_schematic.py
# Thesis output : Figure 5 (fig_2_3_patient_split_schematic), the five-fold cross-validation split schematic
# Inputs        : data/cohort_flow_counts.json; gates read the Chapter 2 LaTeX source under CLAM_TR_PROJECT_DIR; optional fold_facts.json (CLAM_TR_FOLD_FACTS); the thesis build's figure file for the final comparison
# Outputs       : PDF + PNG + previews next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig5_cv_split_schematic.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 5 (label fig:patient_split_schematic; legacy file base name kept so the LaTeX label does not
change): slide-level five-fold cross-validation design of the 9,128-slide noise-cleaned PANDA cohort.

Every number is read from data files; the drawing code contains no count literals.
  * per-fold slide counts, ISUP 0-5 counts, provider counts:  data/cohort_flow_counts.json  panda.cohort_9128
  * near-duplicate groups:                                   data/cohort_flow_counts.json  panda.duplicates
  * configurations (aggregators x HP sets x encoders):        data/cohort_flow_counts.json  panda.trained_runs_crosscheck
Gates (the script refuses to draw when one fails):
  1. fold counts sum to the cohort size; per-fold provider and ISUP counts sum to the fold size;
  2. per-fold ISUP counts summed over folds equal the cohort ISUP counts and Ch.2 Table tab:class_dist;
  3. fold counts equal the five fold nodes of Ch.2 fig:cohort_flow;
  4. (if fold_facts.json is present; CLAM_TR_FOLD_FACTS) fold counts and ISUP counts equal len(slide_ids) and the
     labels of the five fold-indexed prediction files of one experiment on the Drive (fold_facts.json holds these
     values, read from the Drive files; not in this repository); the five slide sets are disjoint and their union
     is the cohort;
  5. no near-duplicate group is split across folds; Ch.2 states the same number of groups;
  6. configuration count from the trained-run inventory equals the count stated in Ch.2;
  7. the Ch.2 text states the stratification variables and the unverifiable patient grouping;
  8. layout: no text outside the canvas, no overlapping text, no text on shapes, smallest font >= MIN_PT (10 pt).
Output (next to this script): PDF at the printed width \textwidth (15.5 cm; the thesis includes it with
\includegraphics[width=\textwidth]), fonts embedded, a PNG twin and 110-dpi previews.

Changes relative to the previous version (same data, gates and texts; KTU thesis guide, section 5.1: figure text
>= 10 pt): drawn at \textwidth instead of 0.85 x 15.5 cm = 13.175 cm; text 7.0/7.5 pt -> 10.0 pt and 8.0 pt ->
10.5 pt (printed = natural); geometry re-spaced; the four notes wrap at the canvas width (line breaks only); the
ISUP legend moved to the second legend row; MIN_PT 10. Added checks: word and number-token multisets identical to
the thesis PDF; every PDF span >= 9.95 pt.

Usage: python -P make_fig5_cv_split_schematic.py
"""
import json
import os  # repo path (reads the environment variables of this repository)
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

PROJECT = Path(os.environ.get("CLAM_TR_PROJECT_DIR", Path(__file__).resolve().parents[2]))  # repo path
R35 = Path(__file__).resolve().parents[1] / "data" / "cohort_flow_counts.json"  # repo path
CH2 = PROJECT / "thesis" / "main" / "bolumler" / "icerik" / "bolum_02_yapilan_calismalar.tex"
HERE = Path(__file__).resolve().parent
DRIVE_FACTS = Path(os.environ.get("CLAM_TR_FOLD_FACTS", Path(__file__).resolve().parents[1] / "data" / "fold_facts.json"))  # repo path (optional input, not in the repository)
OUT_PDF = HERE / "fig_2_3_patient_split_schematic.pdf"
OUT_PNG = HERE / "fig_2_3_patient_split_schematic.png"
INSTALLED = PROJECT / "thesis/main/figures/fig_2_3_patient_split_schematic.pdf"

W_CM = 15.5  # printed width (\textwidth, KTU text width 15.5 cm) = natural width
H_CM = 10.0  # provisional; the final height is fitted to the content in draw()
MIN_PT = 10.0
FS, FS_HEAD = 10.0, 10.5

INK = "#111111"
MUTED = "#4a4a4a"
RULE = "#9a9a9a"
VAL = "#b35a1f"      # validation fold (dark in greyscale)
TRN = "#e6e2da"      # training folds (light)
ISUP_RAMP = ["#e3ebf3", "#bccfe2", "#91aecd", "#6a8db5", "#456b93", "#24476b"]  # ISUP 0 -> 5, light -> dark

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.5,
})


def fmt(x):
    return f"{int(x):,}"


# ----------------------------------------------------------------------------------------------- data + gates
def load_and_gate():
    J = json.loads(R35.read_text(encoding="utf-8"))
    P = J["panda"]
    C = P["cohort_9128"]
    folds = [C["folds"][str(k)] for k in range(len(C["folds"]))]
    n = [f["n"] for f in folds]
    isup = [[f["isup"][str(g)] for g in range(6)] for f in folds]
    rad = [f["radboud"] for f in folds]
    kar = [f["karolinska"] for f in folds]
    total = C["n"]
    isup_total = [C["isup"][str(g)] for g in range(6)]

    # gate 1
    assert len(n) == 5, f"expected five folds, found {len(n)}"
    assert sum(n) == total, (sum(n), total)
    for k in range(5):
        assert rad[k] + kar[k] == n[k], ("provider sum", k)
        assert sum(isup[k]) == n[k], ("ISUP sum", k)
    assert sum(rad) == C["per_provider"]["radboud"] and sum(kar) == C["per_provider"]["karolinska"]

    # gate 2
    assert [sum(isup[k][g] for k in range(5)) for g in range(6)] == isup_total, "per-fold ISUP != cohort ISUP"
    ch2_raw = CH2.read_text(encoding="utf-8")
    ch2 = re.sub(r"\s+", " ", ch2_raw)
    m = re.search(r"Number of slides\s*((?:&\s*[\d,]+\s*)+)\\\\", ch2)
    assert m, "tab:class_dist row not found in Ch.2"
    tab = [int(v.replace(",", "")) for v in re.findall(r"[\d,]+", m.group(1))]
    assert tab == isup_total + [total], ("Ch.2 tab:class_dist", tab)

    # gate 3
    nodes = re.findall(r"Fold (\d) \\\\ (\d+)\{,\}(\d{3})", ch2_raw)
    assert len(nodes) == 5, f"fig:cohort_flow fold nodes not found ({len(nodes)})"
    ch2_n = [None] * 5
    for k, a, b in nodes:
        ch2_n[int(k)] = int(a + b)
    assert ch2_n == n, ("Ch.2 fig:cohort_flow fold counts", ch2_n, n)

    # gate 4 (Drive files of one experiment)
    drive_checked = False
    if DRIVE_FACTS.exists():
        D = json.loads(DRIVE_FACTS.read_text(encoding="utf-8"))
        for k in range(5):
            dk = D["folds"][str(k)]
            assert dk["n"] == n[k], ("Drive fold size", k, dk["n"], n[k])
            assert [dk["isup_counts"][str(g)] for g in range(6)] == isup[k], ("Drive ISUP counts", k)
        assert D["union_unique"] == total and D["pairwise_overlap"] == 0 and D["sum_n"] == total
        drive_checked = True

    # gate 5
    dup = P["duplicates"]
    n_groups = dup["n_groups"]
    assert dup["n_groups_split_across_folds"] == 0
    assert dup["n_groups_all_members_same_fold"] == n_groups
    assert P["denoising"]["kept_kfold_equals_release_kfold"] is True
    assert f"{n_groups} groups" in ch2, "Ch.2 does not state the number of duplicate groups"

    # gate 6
    tr = P["trained_runs_crosscheck"]
    n_cfg = sum(len(tr[e]["aggregators_with_5_bestHP_fold_files"]) + len(tr[e]["aggregators_with_5_defaultHP_fold_files"])
                for e in tr)
    assert f"all {n_cfg} configurations" in ch2, f"Ch.2 does not state 'all {n_cfg} configurations'"
    for e in tr:
        for k, v in tr[e]["retnet_bestHP_folds"].items():
            assert v["equals_csv_fold"] is True and v["n"] == n[int(k)], ("run folds vs CSV", e, k)

    # gate 7
    # the shortened caption no longer carries these statements; they are in the Ch.2 body text (Sec. on
    # slide-level isolation), so the gate checks the body-text wording.
    for phrase in ("stratified on the Gleason score and the data provider", "patient-grouping question",
                   "cannot be verified from the released metadata"):
        assert phrase in ch2, f"Ch.2 phrase missing: {phrase}"
    joint = [ln for ln in dup["handling_in_code"] if "y_gleason" in ln or "y_provider" in ln or "MultilabelStratifiedKFold" in ln]
    assert len(joint) >= 3, "stratification lines not found in the flow-count record"

    return dict(n=n, isup=isup, rad=rad, total=total, n_groups=n_groups, n_cfg=n_cfg, drive_checked=drive_checked)


# ----------------------------------------------------------------------------------------------- drawing
class Canvas:
    def __init__(self, h_cm=H_CM):
        self.fig = plt.figure(figsize=(W_CM / 2.54, h_cm / 2.54))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W_CM)
        self.ax.set_ylim(h_cm, 0)          # y grows downward, units = cm
        self.ax.axis("off")
        self.texts = []
        self.shapes = []

    def text(self, x, y, s, size=7.5, ha="left", va="center", color=INK, weight="normal", style="normal"):
        t = self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, fontweight=weight, fontstyle=style)
        self.texts.append(t)
        return t

    def rect(self, x, y, w, h, fc, ec="none", lw=0.0, z=2, check=True):
        r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
        self.ax.add_patch(r)
        if check:
            self.shapes.append(r)
        return r

    def line(self, xs, ys, color=RULE, lw=0.5):
        self.ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="butt", zorder=1)

    def check_layout(self):
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        fb = self.fig.bbox
        bad = []
        boxes = [(t, t.get_window_extent(rend)) for t in self.texts]
        for t, b in boxes:
            if b.x0 < fb.x0 + 1 or b.x1 > fb.x1 - 1 or b.y0 < fb.y0 + 1 or b.y1 > fb.y1 - 1:
                bad.append(("outside canvas", t.get_text()))
            if t.get_fontsize() < MIN_PT:
                bad.append(("font < MIN_PT", t.get_text()))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                if a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5 and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5:
                    bad.append(("text overlap", boxes[i][0].get_text() + " | " + boxes[j][0].get_text()))
        for t, b in boxes:
            for s in self.shapes:
                sb = s.get_window_extent(rend)
                if b.x0 < sb.x1 - 0.5 and sb.x0 < b.x1 - 0.5 and b.y0 < sb.y1 - 0.5 and sb.y0 < b.y1 - 0.5:
                    bad.append(("text on shape", t.get_text()))
        return bad


NOTE_X, NOTE_W = 0.36, W_CM - 0.36 - 0.10      # note text left edge and maximum width (cm)
LINE_CM = FS * 1.2 / 72 * 2.54                   # matplotlib default line spacing 1.2


def wrap(s, size, max_cm):
    """Break s at spaces only (no hyphenation, so no new tokens) so that every line fits max_cm."""
    f = plt.figure(figsize=(W_CM / 2.54, 3))
    a = f.add_axes([0, 0, 1, 1])
    r = f.canvas.get_renderer()

    def width_cm(txt):
        t = a.text(0, 0, txt, fontsize=size)
        w = t.get_window_extent(r).width / f.dpi * 2.54
        t.remove()
        return w

    out, cur = [], ""
    for w in s.split(" "):
        trial = w if not cur else cur + " " + w
        if width_cm(trial) <= max_cm:
            cur = trial
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    plt.close(f)
    assert " ".join(out) == s
    return out


def draw(d):
    n, isup, rad = d["n"], d["isup"], d["rad"]
    notes = [
        f"Each slide belongs to exactly one fold, and none of the {d['n_groups']} near-duplicate image groups spans two folds.",
        "Folds stratified on Gleason score × data provider and assigned before label denoising.",
        "No patient identifiers in PANDA: the split is at slide level, and patient grouping cannot be verified.",
        f"The same five folds serve all {d['n_cfg']} configurations, so fold-level results can be compared pairwise.",
    ]
    wrapped = [wrap(s, FS, NOTE_W) for s in notes]

    # geometry (cm)
    X_LAB = 1.30                 # right edge of the fold labels
    X_GRID, CW, CG = 1.50, 0.98, 0.11
    Y0, RH, RG = 1.30, 0.62, 0.22
    X_BAR, SCALE = 7.05, 4.6 / 2000.0      # 2,000 slides = 4.6 cm
    X_RAD = 14.45
    rows_y = [Y0 + k * (RH + RG) for k in range(5)]
    grid_mid = X_GRID + (5 * CW + 4 * CG) / 2
    bar_axis_end = X_BAR + 2000 * SCALE
    y_ax = rows_y[-1] + RH + 0.16
    y_l1 = y_ax + 1.05
    y_l2 = y_l1 + 0.55
    y_n = y_l2 + 0.50                      # top of the first note
    note_tops = []
    y = y_n
    for wl in wrapped:
        note_tops.append(y)
        y += len(wl) * LINE_CM + 0.12
    h_cm = round(y - 0.12 + 0.15, 3)

    c = Canvas(h_cm)
    # headers
    c.text(grid_mid, 0.38, "Cross-validation round", size=FS_HEAD, ha="center", weight="bold")
    for r in range(5):
        c.text(X_GRID + r * (CW + CG) + CW / 2, 0.93, str(r), size=FS, ha="center", color=MUTED)
    c.text(X_BAR, 0.38, "Slides per fold by ISUP grade", size=FS_HEAD, ha="left", weight="bold")
    c.text(X_BAR, 0.93, f"({fmt(d['total'])} in total)", size=FS, ha="left", color=MUTED)
    c.text(X_RAD, 0.38, "Radboud", size=FS_HEAD, ha="center", weight="bold")
    c.text(X_RAD, 0.93, "share", size=FS, ha="center", color=MUTED)

    # rows
    bar_right = []
    for k in range(5):
        y = rows_y[k]
        c.text(X_LAB, y + RH / 2, f"Fold {k}", size=FS_HEAD, ha="right")
        for r in range(5):
            c.rect(X_GRID + r * (CW + CG), y, CW, RH, VAL if r == k else TRN)
        x = X_BAR
        for g in range(6):
            w = isup[k][g] * SCALE
            c.rect(x, y, w, RH, ISUP_RAMP[g], ec="white", lw=0.35, z=2)
            x += w
        bar_right.append(x)
        c.text(x + 0.12, y + RH / 2, fmt(n[k]), size=FS, ha="left")
        c.text(X_RAD, y + RH / 2, f"{100.0 * rad[k] / n[k]:.1f}%", size=FS, ha="center")
    for k in range(5):                     # rendered bar length = fold size (data-drawing identity)
        assert abs((bar_right[k] - X_BAR) / SCALE - n[k]) < 1e-6, ("bar length", k)

    # bar axis
    c.line([X_BAR, bar_axis_end], [y_ax, y_ax], color=RULE, lw=0.5)
    for v in range(0, 2001, 500):
        xv = X_BAR + v * SCALE
        c.line([xv, xv], [y_ax, y_ax + 0.08], color=RULE, lw=0.5)
        c.text(xv, y_ax + 0.36, fmt(v), size=FS, ha="center", color=MUTED)
    c.text(bar_axis_end + 0.55, y_ax + 0.36, "slides", size=FS, ha="left", color=MUTED)

    # legends: row 1 validation fold; row 2 training folds + ISUP grade ramp
    sw = 0.32
    c.rect(0.06, y_l1 - sw / 2, sw * 1.4, sw, VAL)
    c.text(0.06 + sw * 1.4 + 0.14, y_l1, "validation fold: early stopping and reported QWK", size=FS)
    c.rect(0.06, y_l2 - sw / 2, sw * 1.4, sw, TRN)
    c.text(0.06 + sw * 1.4 + 0.14, y_l2, "training folds (the other four)", size=FS)
    c.text(X_BAR, y_l2, "ISUP grade", size=FS, ha="left")
    xs = X_BAR + 1.85
    for g in range(6):
        c.rect(xs + g * 0.78, y_l2 - sw / 2, sw * 1.3, sw, ISUP_RAMP[g], ec=RULE, lw=0.3)
        c.text(xs + g * 0.78 + sw * 1.3 + 0.08, y_l2, str(g), size=FS, ha="left")

    # notes (wrapped at spaces only)
    for top, wl in zip(note_tops, wrapped):
        c.text(0.06, top, "•", size=FS, ha="left", va="top", color=MUTED)
        c.text(NOTE_X, top, "\n".join(wl), size=FS, ha="left", va="top")

    bad = c.check_layout()
    if bad:
        for b in bad:
            print("LAYOUT", b)
        raise SystemExit(f"{len(bad)} layout problems; figure not saved")
    sizes = sorted({t.get_fontsize() for t in c.texts})
    c.fig.savefig(OUT_PDF, metadata={"CreationDate": None, "ModDate": None})
    c.fig.savefig(OUT_PNG, dpi=600)
    plt.close(c.fig)
    print(f"saved {OUT_PDF} + PNG  ({W_CM} x {h_cm} cm; font sizes used {sizes}; {len(c.texts)} texts, "
          f"{len(c.shapes)} shapes checked; note lines {[len(w) for w in wrapped]})")


def check_pdf():
    import collections
    import statistics

    import fitz

    fails = []
    new, old = fitz.open(OUT_PDF)[0], fitz.open(INSTALLED)[0]
    TW_BP = 441.01773 * 72 / 72.27
    scale = TW_BP / new.rect.width
    sp = [s for b in new.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
    psz = [s["size"] * scale for s in sp]
    small = [(s["text"], round(s["size"] * scale, 2)) for s in sp if s["size"] * scale < 9.95]
    print(f"  page {new.rect.width:.2f} x {new.rect.height:.2f} bp; scale at \\textwidth {scale:.4f}; printed "
          f"{new.rect.width * scale / 72 * 2.54:.2f} x {new.rect.height * scale / 72 * 2.54:.2f} cm")
    print(f"  printed size min {min(psz):.2f} median {statistics.median(psz):.2f}; fonts {sorted({s['font'] for s in sp})}")
    if small:
        fails.append(f"spans < 9.95 pt: {small}")
    ot, nt = old.get_text(), new.get_text()
    ow, nw = collections.Counter(ot.split()), collections.Counter(nt.split())
    print(f"  words: {'identical' if ow == nw else 'DIFFERENT'} ({sum(ow.values())})")
    if ow != nw:
        fails.append(f"words old-only {dict(ow - nw)} new-only {dict(nw - ow)}")
    NUM = re.compile(r"(?<![A-Za-z0-9])[+\-−–]?\d+(?:[.,]\d+)*")
    on, nn = collections.Counter(NUM.findall(ot)), collections.Counter(NUM.findall(nt))
    print(f"  number tokens: {'identical' if on == nn else 'DIFFERENT'} ({sum(on.values())})")
    if on != nn:
        fails.append(f"numbers old-only {dict(on - nn)} new-only {dict(nn - on)}")
    outside = [b[4][:30] for b in new.get_text("blocks") if b[0] < 0 or b[1] < 0 or b[2] > new.rect.width or b[3] > new.rect.height]
    if outside:
        fails.append(f"text outside page {outside}")
    z = 110 / 72
    pm = new.get_pixmap(matrix=fitz.Matrix(z, z))
    pm.set_dpi(110, 110)
    pm.save(str(HERE / "preview_fig_2_3_patient_split_schematic_110dpi.png"))
    pm = new.get_pixmap(dpi=200)
    pm.save(str(HERE / "preview_fig_2_3_patient_split_schematic_print200dpi.png"))
    print("ALL CHECKS PASSED" if not fails else "FAILED:\n  " + "\n  ".join(fails))


def main():
    d = load_and_gate()
    print("gates passed: fold sizes", [fmt(v) for v in d["n"]], "sum", fmt(sum(d["n"])),
          "| duplicate groups", d["n_groups"], "| configurations", d["n_cfg"],
          "| Drive cross-check", "done" if d["drive_checked"] else "SKIPPED (fold_facts.json absent)")
    draw(d)
    check_pdf()


if __name__ == "__main__":
    main()

