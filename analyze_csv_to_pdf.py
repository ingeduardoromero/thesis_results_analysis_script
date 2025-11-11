#!/usr/bin/env python3
"""
Analyze a CSV and output a PDF report.

Modes:
- Generic CSV summary (existing behavior)
- Likert survey analysis with role breakdowns, composites, and reliability

Usage:
  python analyze_csv_to_pdf.py "input.csv" ["output.pdf"]

No external dependencies required. The PDF contains text-only pages.
"""

import csv
import math
import os
import sys
import statistics
import textwrap
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional plotting (matplotlib). If unavailable, we fallback to text-only PDF.
HAS_MPL = False
try:
    import matplotlib  # type: ignore
    matplotlib.use('Agg')  # force non-GUI backend for headless environments
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------
# Utilities for type detection
# -----------------------------

def _is_missing(value: str) -> bool:
    return value is None or str(value).strip() == ""


def _parse_number(raw: str) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    # Handle accounting negatives (e.g., (123.45))
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    # Remove common decorations
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    # Handle trailing/leading + and -
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


def _coerce_latin1(s: str) -> str:
    """Coerce text to Latin-1 for basic PDF Type1 font compatibility.
    Non-encodable characters are replaced with '?' to avoid PDF errors.
    """
    try:
        return s.encode("latin-1", errors="replace").decode("latin-1")
    except Exception:
        return s


# -----------------------------
# CSV analysis
# -----------------------------

def analyze_csv(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = [r for r in reader]

    total_rows = len(rows)
    total_cols = len(fieldnames)

    # Prepare per-column value lists
    col_values: Dict[str, List[str]] = {col: [] for col in fieldnames}
    missing_cells = 0
    total_cells = total_rows * max(total_cols, 1)

    for r in rows:
        for col in fieldnames:
            val = r.get(col, "")
            if _is_missing(val):
                missing_cells += 1
            else:
                col_values[col].append(str(val))

    columns_summary: List[Dict[str, Any]] = []

    for col in fieldnames:
        values = col_values[col]
        non_missing_count = len(values)
        missing_count = total_rows - non_missing_count
        numeric_values: List[float] = []
        for v in values:
            num = _parse_number(v)
            if num is not None and math.isfinite(num):
                numeric_values.append(num)

        numeric_ratio = (len(numeric_values) / non_missing_count) if non_missing_count else 0.0
        col_summary: Dict[str, Any] = {
            "name": col,
            "non_missing": non_missing_count,
            "missing": missing_count,
            "missing_pct": (missing_count / total_rows * 100.0) if total_rows else 0.0,
        }

        if non_missing_count == 0:
            col_summary["type"] = "empty"
        elif numeric_ratio >= 0.7:  # treat as numeric when most values parse as numbers
            nums = numeric_values
            col_summary["type"] = "numeric"
            if len(nums) >= 1:
                col_summary["count"] = len(nums)
                col_summary["mean"] = statistics.fmean(nums)
                col_summary["median"] = statistics.median(nums)
                col_summary["stdev"] = statistics.pstdev(nums) if len(nums) > 1 else 0.0
                col_summary["min"] = min(nums)
                col_summary["max"] = max(nums)
            else:
                col_summary["count"] = 0
        else:
            # categorical/text summary
            col_summary["type"] = "categorical"
            counter = Counter([v.strip() for v in values if v.strip() != ""])
            col_summary["unique"] = len(counter)
            col_summary["top"] = counter.most_common(5)

        columns_summary.append(col_summary)

    return {
        "path": path,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": total_rows,
        "cols": total_cols,
        "missing_cells": missing_cells,
        "missing_pct": (missing_cells / total_cells * 100.0) if total_cells else 0.0,
        "columns": columns_summary,
    }


# -----------------------------
# Likert (1-5) survey analysis
# -----------------------------

def _find_role_column(fieldnames: List[str]) -> Optional[str]:
    for col in fieldnames:
        key = col.strip().strip(':').lower()
        if key == "role" or key.endswith(" role") or key.startswith("role"):
            return col
    return None


def _find_sector_column(fieldnames: List[str]) -> Optional[str]:
    for col in fieldnames:
        key = col.strip().strip(':').lower()
        if "sector" in key or key == "industry" or "regulated" in key:
            return col
    return None


def _int_or_none(x: Any) -> Optional[int]:
    try:
        s = str(x).strip()
        if s == "":
            return None
        v = int(float(s))
        if 1 <= v <= 5:
            return v
        return None
    except Exception:
        return None


def _cronbach_alpha(items: List[List[Optional[int]]]) -> Optional[float]:
    # items: list of item response lists aligned by respondent index
    # compute alpha over respondents with complete data for the set
    if not items or len(items) < 2:
        return None
    # collect rows with no missing
    n = len(items[0])
    complete_rows: List[List[float]] = []
    for i in range(n):
        row = []
        missing = False
        for it in items:
            v = it[i]
            if v is None:
                missing = True
                break
            row.append(float(v))
        if not missing:
            complete_rows.append(row)
    if len(complete_rows) < 2:
        return None
    k = len(items)
    # variances
    def var(vals: List[float]) -> float:
        m = statistics.fmean(vals)
        return statistics.pvariance(vals, mu=m)
    item_vars = [var([row[j] for row in complete_rows]) for j in range(k)]
    totals = [sum(row) for row in complete_rows]
    total_var = var(totals)
    if total_var == 0:
        return None
    alpha = (k / (k - 1)) * (1 - sum(item_vars) / total_var)
    return alpha


def _rank_avg(vals: List[float]) -> List[float]:
    # average ranks for ties, starting at 1
    idx = list(range(len(vals)))
    idx.sort(key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    xr = _rank_avg(xvals)
    yr = _rank_avg(yvals)
    mx = statistics.fmean(xr)
    my = statistics.fmean(yr)
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xr))
    deny = math.sqrt(sum((b - my) ** 2 for b in yr))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def analyze_likert_csv(path: str) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = [r for r in reader]

    if not fieldnames or not rows:
        return None

    role_col = _find_role_column(fieldnames)
    sector_col = _find_sector_column(fieldnames)
    # if no role column, continue but mark None
    item_cols = [c for c in fieldnames if c != role_col and c != sector_col]
    if len(item_cols) < 3:
        return None

    # Map item columns to Q1..Qn by order, try to read numeric prefix if present
    item_ids: List[str] = []
    item_labels: Dict[str, str] = {}
    for idx, col in enumerate(item_cols, start=1):
        label = col.strip()
        # Try numeric prefix like "1. ..."
        qid = None
        head = label.split(".", 1)[0].strip()
        if head.isdigit():
            qid = f"Q{int(head)}"
        else:
            qid = f"Q{idx}"
        item_ids.append(qid)
        item_labels[qid] = label

    # Collect responses
    role_values: List[Optional[str]] = []
    sector_values: List[Optional[str]] = []
    item_responses: Dict[str, List[Optional[int]]] = {qid: [] for qid in item_ids}
    for r in rows:
        role_values.append(r.get(role_col) if role_col else None)
        sector_values.append(r.get(sector_col) if sector_col else None)
        for qid, col in zip(item_ids, item_cols):
            item_responses[qid].append(_int_or_none(r.get(col, "")))

    n = len(rows)

    # Per-item stats
    per_item = []
    role_set = sorted({rv for rv in role_values if rv and str(rv).strip() != ""})
    # Role counts for cover-page pie chart
    role_counts = Counter()
    for rv in role_values:
        if rv and str(rv).strip() != "":
            role_counts[str(rv).strip()] += 1
    sector_set = sorted({sv for sv in sector_values if sv and str(sv).strip() != ""})
    # Sector counts for cover-page pie chart
    sector_counts = Counter()
    for sv in sector_values:
        if sv and str(sv).strip() != "":
            sector_counts[str(sv).strip()] += 1
    for qid in item_ids:
        vals = item_responses[qid]
        present = [v for v in vals if v is not None]
        counts = {k: 0 for k in [1, 2, 3, 4, 5]}
        for v in present:
            if 1 <= v <= 5:
                counts[v] += 1
        total = len(present)
        mean = statistics.fmean(present) if total else None
        median = statistics.median(present) if total else None
        stdev = statistics.pstdev(present) if total > 1 else 0.0 if total == 1 else None
        disagree = counts[1] + counts[2]
        neutral = counts[3]
        agree = counts[4] + counts[5]
        floor_flag = (counts[1] / total >= 0.4) if total else False
        ceiling_flag = (counts[5] / total >= 0.4) if total else False
        # role means
        role_means: Dict[str, Optional[float]] = {}
        if role_col:
            for role in role_set:
                p = [v for v, rv in zip(vals, role_values) if rv == role and v is not None]
                role_means[role] = statistics.fmean(p) if p else None
        # sector distributions
        sector_pcts: Dict[str, Dict[int, float]] = {}
        if sector_col:
            for sector in sector_set:
                sv = [v for v, svv in zip(vals, sector_values) if svv == sector and v is not None]
                counts_s = {k: 0 for k in [1, 2, 3, 4, 5]}
                for v in sv:
                    if 1 <= v <= 5:
                        counts_s[v] += 1
                tot_s = len(sv)
                sector_pcts[sector] = {k: (counts_s[k] / tot_s * 100.0) if tot_s else 0.0 for k in counts_s}

        per_item.append({
            "id": qid,
            "label": item_labels[qid],
            "mean": mean,
            "median": median,
            "stdev": stdev,
            "counts": counts,
            "pct": {k: (counts[k] / total * 100.0) if total else 0.0 for k in counts},
            "bands": {
                "disagree_pct": (disagree / total * 100.0) if total else 0.0,
                "neutral_pct": (neutral / total * 100.0) if total else 0.0,
                "agree_pct": (agree / total * 100.0) if total else 0.0,
            },
            "missing": n - total,
            "floor": floor_flag,
            "ceiling": ceiling_flag,
            "role_means": role_means,
            "sector_pct": sector_pcts,
        })

    # Role deltas (largest differences in means between roles) per item
    role_deltas = []
    if role_col and len(role_set) >= 2:
        for item in per_item:
            rm = item["role_means"]
            pairs = []
            roles = list(role_set)
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    a, b = roles[i], roles[j]
                    va, vb = rm.get(a), rm.get(b)
                    if va is not None and vb is not None:
                        pairs.append(((a, b), abs(va - vb), va, vb))
            if pairs:
                top = max(pairs, key=lambda x: x[1])
                role_deltas.append({
                    "item": item["id"],
                    "roles": top[0],
                    "delta": top[1],
                    "means": (top[2], top[3]),
                })
        # top 5 deltas overall
        role_deltas.sort(key=lambda d: d["delta"], reverse=True)
        top_deltas = role_deltas[:5]
    else:
        top_deltas = []

    # Composites and reverse coding
    # Reverse(x) = 6 - x
    def reverse_vec(vs: List[Optional[int]]) -> List[Optional[int]]:
        return [6 - v if v is not None else None for v in vs]

    # Build vectors
    Q = item_responses  # shorthand
    def get(q: str) -> List[Optional[int]]:
        return Q.get(q, [None] * n)

    # Composites spec with reverse-coding where item wording runs opposite to construct
    composites_spec = [
        {"name": "Role Overlap Index", "items": ["Q1", "Q2", "Q3", "Q5", "Q10"], "reverse": []},
        {"name": "Communication Clarity Index", "items": ["Q6", "Q7", "Q9", "Q8"], "reverse": ["Q8"]},
        {"name": "Governance Maturity Index", "items": ["Q4", "Q11", "Q13", "Q14", "Q15"], "reverse": []},
        {"name": "Accountability Ambiguity Index", "items": ["Q2", "Q12", "Q11", "Q15"], "reverse": ["Q11", "Q15"]},
    ]

    composites = []
    # respondent-level composite scores
    for spec in composites_spec:
        item_vecs: List[List[Optional[int]]] = []
        for qid in spec["items"]:
            vec = get(qid)
            if qid in spec.get("reverse", []):
                vec = reverse_vec(vec)
            item_vecs.append(vec)
        # alpha
        alpha = _cronbach_alpha(item_vecs)
        # respondent means with at least half non-missing
        scores: List[Optional[float]] = []
        for i in range(n):
            vals = [vec[i] for vec in item_vecs if vec[i] is not None]
            if len(vals) >= max(1, len(item_vecs) // 2):
                scores.append(statistics.fmean([float(v) for v in vals]))
            else:
                scores.append(None)
        present_scores = [s for s in scores if s is not None]
        comp_mean = statistics.fmean(present_scores) if present_scores else None
        comp_sd = statistics.pstdev(present_scores) if len(present_scores) > 1 else (0.0 if len(present_scores) == 1 else None)
        # role means
        comp_role_means: Dict[str, Optional[float]] = {}
        if role_col:
            for role in role_set:
                vals = [s for s, rv in zip(scores, role_values) if rv == role and s is not None]
                comp_role_means[role] = statistics.fmean(vals) if vals else None
        comp_sector_means: Dict[str, Optional[float]] = {}
        if sector_col:
            for sector in sector_set:
                vals = [s for s, sv in zip(scores, sector_values) if sv == sector and s is not None]
                comp_sector_means[sector] = statistics.fmean(vals) if vals else None

        composites.append({
            "name": spec["name"],
            "items": spec["items"],
            "reverse": spec.get("reverse", []),
            "alpha": alpha,
            "mean": comp_mean,
            "stdev": comp_sd,
            "role_means": comp_role_means,
            "sector_means": comp_sector_means,
            "scores": scores,
        })

    # Governance vs Ambiguity gap and quadrants
    def comp_by_name(name: str) -> Optional[Dict[str, Any]]:
        for c in composites:
            if c["name"] == name:
                return c
        return None

    gov = comp_by_name("Governance Maturity Index")
    amb = comp_by_name("Accountability Ambiguity Index")
    gap = None
    if gov and amb:
        def pct_agree(spec_items: List[str], rev: List[str]) -> float:
            flags: List[bool] = []
            for i in range(n):
                hit = False
                for q in spec_items:
                    val = get(q)[i]
                    if q in rev and val is not None:
                        val = 6 - val
                    if val is not None and val >= 4:
                        hit = True
                        break
                flags.append(hit)
            if not flags:
                return 0.0
            return sum(1 for f in flags if f) / len(flags) * 100.0

        gov_pct = pct_agree(gov["items"], gov["reverse"]) if gov else 0.0
        amb_pct = pct_agree(amb["items"], amb["reverse"]) if amb else 0.0

        # Quadrants by mean thresholds (>=4.0 high)
        high_gov = [s is not None and s >= 4.0 for s in gov["scores"]]
        high_amb = [s is not None and s >= 4.0 for s in amb["scores"]]
        counts = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
        for hg, ha in zip(high_gov, high_amb):
            if hg is False and ha is False:
                counts["LL"] += 1
            elif hg is True and ha is False:
                counts["HL"] += 1
            elif hg is False and ha is True:
                counts["LH"] += 1
            elif hg is True and ha is True:
                counts["HH"] += 1
        gap = {"gov_agree_pct": gov_pct, "amb_agree_pct": amb_pct, "quadrants": counts}

    # Associations (selected Spearman correlations)
    correlations = []
    if gov and amb:
        pairs = [
            ("Role Overlap vs Governance", comp_by_name("Role Overlap Index"), gov),
            ("Ambiguity vs Governance", amb, gov),
            ("Communication vs Overlap", comp_by_name("Communication Clarity Index"), comp_by_name("Role Overlap Index")),
        ]
        for name, a, b in pairs:
            if a and b:
                rho = _spearman(a["scores"], b["scores"])  # type: ignore[arg-type]
                correlations.append({"name": name, "rho": rho})

    return {
        "mode": "likert",
        "path": path,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": n,
        "role_col": role_col,
        "roles": role_set,
        "role_counts": dict(role_counts),
        "sector_col": sector_col,
        "sectors": sector_set,
        "sector_counts": dict(sector_counts),
        "items": per_item,
        "top_role_deltas": top_deltas,
        "composites": composites,
        "gap": gap,
        "correlations": correlations,
    }


# -----------------------------
# Matplotlib rendering (stacked bars & charts)
# -----------------------------

def _likert_colors():
    # Colorblind-friendly palette for 1..5
    # 1: Strongly Disagree -> darker red, 5: Strongly Agree -> darker blue/green
    return [
        (178/255, 34/255, 34/255),    # 1 firebrick
        (239/255, 138/255, 98/255),   # 2 salmon-like
        (224/255, 224/255, 224/255),  # 3 grey
        (103/255, 169/255, 207/255),  # 4 light blue
        (33/255, 113/255, 181/255),   # 5 blue
    ]


def _draw_likert_stacked(
    ax,
    groups: List[Tuple[str, Dict[int, float]]],
    title: str = "",
    show_legend: bool = True,
    title_fs: int = 11,
):
    # Draw vertical stacked bars (x = groups, y = % stacked 1..5)
    colors = _likert_colors()
    labels = [str(g[0]) for g in groups]
    data = [[g[1].get(i, 0.0) for i in [1, 2, 3, 4, 5]] for g in groups]
    n = max(1, len(groups))
    # Tight packing: explicit x positions with small inter-bar gap
    # Make bars very slim so they nearly match label width
    width = 0.035 if n <= 6 else 0.030
    # Keep a slightly larger inter-bar gap to prevent visual merging
    gap = max(0.14, width * 3.5)
    x = [i * (width + gap) for i in range(n)]
    bottom = [0.0] * n
    # Decide label strategy: 1–5 in-bar labels with threshold; fallback to 3-band labels if too many tiny slices
    label_threshold = 6.0  # percent
    decimals = 0
    visible_slices = 0
    for i in range(5):
        visible_slices += sum(1 for row in data if row[i] >= label_threshold)

    use_band_labels = (visible_slices / max(1, n)) < 2  # fewer than ~2 visible slices per bar on average

    def text_color_for_bg(rgb):
        r, g, b = rgb
        # relative luminance approximation
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return 'black' if y > 0.6 else 'white'

    for i in range(5):
        vals = [row[i] for row in data]
        prev_bottom = bottom[:]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=width,
            color=colors[i],
            edgecolor='white',
            linewidth=0.5,
            label=str(i + 1),
        )
        # 1–5 labels if not using band labels
        if not use_band_labels:
            for xi, v, btm in zip(x, vals, prev_bottom):
                if v >= label_threshold:
                    ax.text(
                        xi,
                        btm + v / 2.0,
                        f"{v:.{decimals}f}%",
                        ha='center',
                        va='center',
                        fontsize=8,
                        color=text_color_for_bg(colors[i]),
                    )
        bottom = [b + v for b, v in zip(prev_bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of responses')
    if title:
        ax.set_title(title, fontsize=title_fs)
    # Legend optional; caller may draw a shared legend at figure level
    if show_legend:
        ax.legend(title='Scale', loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False, fontsize=9, title_fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    # Tight x-limits and margins to reduce outside whitespace
    if n >= 1:
        ax.set_xlim(x[0] - width * 0.6, x[-1] + width * 1.0)
    ax.margins(x=0)

    # If using band labels, draw D/N/A text centered in each block
    if use_band_labels:
        for xi, row in zip(x, data):
            d = row[0] + row[1]
            ntr = row[2]
            a = row[3] + row[4]
            # positions relative to bottom=0..100
            if d >= label_threshold:
                ax.text(xi, d / 2.0, f"{d:.{decimals}f}%", ha='center', va='center', fontsize=8, color='white')
            if ntr >= label_threshold:
                ax.text(xi, d + ntr / 2.0, f"{ntr:.{decimals}f}%", ha='center', va='center', fontsize=8, color='black')
            if a >= label_threshold:
                ax.text(xi, d + ntr + a / 2.0, f"{a:.{decimals}f}%", ha='center', va='center', fontsize=8, color='white')


def render_pdf_with_matplotlib(summary: Dict[str, Any], output_path: str, export_prefix: Optional[str] = None) -> None:
    if summary.get('mode') != 'likert':
        # Fallback: plain PDF layout drawn via matplotlib text
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            txt = [
                'CSV Analysis Report',
                '',
                f"File: {os.path.basename(summary['path'])}",
                f"Generated: {summary['generated_at']}",
                '',
                f"Rows: {summary['rows']}",
                f"Columns: {summary['cols']}",
                f"Missing cells: {_fmt_float(summary['missing_pct'],2)}%",
                '',
                'Columns:',
            ]
            y = 1.0
            for i, line in enumerate(txt):
                ax.text(0.05, y, line, transform=ax.transAxes, fontsize=12 - (2 if i>0 else 0), va='top')
                y -= 0.05
            for col in summary['columns'][:20]:
                ax.text(0.07, y, f"- {col['name']} ({col['type']})", transform=ax.transAxes, fontsize=9, va='top')
                y -= 0.03
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        return

    roles = summary.get('roles', [])
    sectors = summary.get('sectors', [])
    sector_col = summary.get('sector_col')

    with PdfPages(output_path) as pdf:
        scale_desc = (
            "Scale: 1=Strongly Disagree • 2=Disagree • 3=Neutral • 4=Agree • 5=Strongly Agree"
        )
        # Cover / Overview
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        # Build basic cover lines (concise)
        lines = [
            'Survey Analysis Report',
            f"File: {os.path.basename(summary['path'])}",
            f"Generated: {summary['generated_at']}",
            '',
            f"Responses: {summary['rows']}",
            f"Unique roles: {len(roles) if roles else 0}",
            f"Unique sectors: {len(sectors) if sectors else 0}",
        ]
        # Add question count only here; category mapping will be shown in a small panel below the bar chart
        try:
            q_count = len(summary.get('items', []))
            if q_count:
                lines += [f"Questions: {q_count}"]
        except Exception:
            pass
        y = 0.95
        for i, line in enumerate(lines):
            fs = 18 if i == 0 else 11
            ax.text(0.06, y, line, transform=ax.transAxes, fontsize=fs, va='top')
            y -= 0.05
        # Replace verbose lists with pies and concise counts (see right side pies)
        # Cover pie chart: respondents by role
        rc = summary.get('role_counts') or {}
        if rc:
            items = sorted(rc.items(), key=lambda kv: kv[1], reverse=True)
            if len(items) > 6:
                top = items[:5]
                other = sum(v for _, v in items[5:])
                top.append(('Other', other))
            else:
                top = items
            labels = [k for k, _ in top]
            sizes = [v for _, v in top]
            ax_pie = fig.add_axes([0.55, 0.55, 0.38, 0.38])
            colors = list(plt.cm.tab20.colors)
            wedges, texts, autotexts = ax_pie.pie(
                sizes,
                startangle=90,
                autopct='%1.0f%%',
                counterclock=False,
                colors=colors[:len(sizes)],
                textprops={'fontsize': 9}
            )
            ax_pie.axis('equal')
            ax_pie.set_title('Respondents by Role', fontsize=11)
            legend_labels = [f"{lab} (n={cnt})" for lab, cnt in zip(labels, sizes)]
            ax_pie.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=9, ncol=1, frameon=False)
            # Caption under role pie
            pos_role = ax_pie.get_position()
            fig.text((pos_role.x0 + pos_role.x1) / 2.0, pos_role.y0 - 0.02,
                     'Figure 1. Respondents by role (pie).',
                     ha='center', va='top', fontsize=9, color='dimgray')
            # Export standalone role pie
            if export_prefix:
                try:
                    fig_r, ax_r = plt.subplots(figsize=(5, 5))
                    wr, _, _ = ax_r.pie(
                        sizes,
                        startangle=90,
                        autopct='%1.0f%%',
                        counterclock=False,
                        colors=colors[:len(sizes)],
                        textprops={'fontsize': 10}
                    )
                    ax_r.axis('equal')
                    ax_r.set_title('Respondents by Role', fontsize=12)
                    ax_r.legend(wr, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=1, frameon=False)
                    fig_r.savefig(f"{export_prefix}_figure1_role_pie.png", dpi=200, bbox_inches='tight')
                    plt.close(fig_r)
                except Exception:
                    pass
        # Cover pie chart: respondents by sector
        sc = summary.get('sector_counts') or {}
        if sc:
            items = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
            if len(items) > 6:
                top = items[:5]
                other = sum(v for _, v in items[5:])
                top.append(('Other', other))
            else:
                top = items
            labels = [k for k, _ in top]
            sizes = [v for _, v in top]
            ax_pie2 = fig.add_axes([0.55, 0.12, 0.38, 0.32])
            colors2 = list(plt.cm.Set3.colors)
            wedges2, texts2, autotexts2 = ax_pie2.pie(
                sizes,
                startangle=90,
                autopct='%1.0f%%',
                counterclock=False,
                colors=colors2[:len(sizes)],
                textprops={'fontsize': 9}
            )
            ax_pie2.axis('equal')
            ax_pie2.set_title('Respondents by Sector', fontsize=11)
            legend_labels2 = [f"{lab} (n={cnt})" for lab, cnt in zip(labels, sizes)]
            ax_pie2.legend(wedges2, legend_labels2, loc='lower center', bbox_to_anchor=(0.5, -0.28), fontsize=9, ncol=1, frameon=False)
            # Caption under sector pie
            pos2 = ax_pie2.get_position()
            fig.text((pos2.x0 + pos2.x1) / 2.0, pos2.y0 - 0.065,
                     'Figure 2. Respondents by sector (pie).',
                     ha='center', va='top', fontsize=9, color='dimgray')
            # Export standalone sector pie
            if export_prefix:
                try:
                    fig_s, ax_s = plt.subplots(figsize=(5, 5))
                    ws, _, _ = ax_s.pie(
                        sizes,
                        startangle=90,
                        autopct='%1.0f%%',
                        counterclock=False,
                        colors=colors2[:len(sizes)],
                        textprops={'fontsize': 10}
                    )
                    ax_s.axis('equal')
                    ax_s.set_title('Respondents by Sector', fontsize=12)
                    ax_s.legend(ws, legend_labels2, loc='lower center', bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=1, frameon=False)
                    fig_s.savefig(f"{export_prefix}_figure2_sector_pie.png", dpi=200, bbox_inches='tight')
                    plt.close(fig_s)
                except Exception:
                    pass

        # Cover bar chart: number of items per category (composite)
        comps = summary.get('composites') or []
        if comps:
            # Position bar chart between header lines and mapping text
            ax_bar = fig.add_axes([0.08, 0.24, 0.38, 0.26])
            names = [c['name'] for c in comps]
            counts = [len(c.get('items', [])) for c in comps]
            x = list(range(len(names)))
            ax_bar.bar(x, counts, color='#6baed6', width=0.5)
            ax_bar.set_xticks(x)
            # Shorten names for x labels
            short = []
            for n in names:
                key = n.lower()
                if 'overlap' in key:
                    short.append('Overlap')
                elif 'communication' in key:
                    short.append('Comm')
                elif 'governance' in key:
                    short.append('Gov')
                elif 'ambiguity' in key:
                    short.append('Ambiguity')
                else:
                    short.append(n.replace(' Index',''))
            ax_bar.set_xticklabels(short, rotation=0, ha='center', fontsize=9)
            ax_bar.set_ylabel('Items', fontsize=10)
            ax_bar.set_title('Questions per Category', fontsize=10, pad=8)
            ax_bar.set_ylim(0, max(counts) + 1)
            for xi, c in zip(x, counts):
                ax_bar.text(xi, c + 0.05, str(c), ha='center', va='bottom', fontsize=9)
            # Caption under bar chart
            bpos = ax_bar.get_position()
            fig.text((bpos.x0 + bpos.x1) / 2.0, bpos.y0 - 0.02,
                     '',
                     ha='center', va='top', fontsize=9, color='dimgray')
            # Category mapping text panel below the bar chart
            try:
                # Place mapping slightly below the bar chart caption and include counts
                mapping_ax = fig.add_axes([0.08, 0.06, 0.38, 0.16])
                mapping_ax.axis('off')
                lines_map = ["Categories:"]
                for c, cnt in zip(comps, counts):
                    rev = set(c.get('reverse') or [])
                    items_lbls = [it + ('*' if it in rev else '') for it in c.get('items', [])]
                    # Abbreviate long names
                    cname = c.get('name','').replace(' Index','')
                    lines_map.append(f"- {cname} ({cnt}): {', '.join(items_lbls)}")
                lines_map.append('(*) reverse-coded')
                mapping_ax.text(0, 1, "\n".join(lines_map), va='top', fontsize=9)
            except Exception:
                pass
        # Save cover without Likert scale caption to reduce clutter
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Per-question stacked bars (Overall + per sector) with question text on the left
        # Fit up to 3 questions per page for better space usage
        max_groups = 1 + min(len(sectors), 4)  # Overall + up to 4 sectors
        items = summary['items']
        for i in range(0, len(items), 3):
            chunk = items[i:i+3]
            rows = len(chunk)
            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(rows, 2, width_ratios=[1.0, 1.3], hspace=0.55, wspace=0.3)
            for row_idx, item in enumerate(chunk):
                # Left column: full question text, wrapped
                txt_ax = fig.add_subplot(gs[row_idx, 0])
                txt_ax.axis('off')
                q_text = f"{item['id']}: {item['label']}"
                # Aim for ~3 lines: approximate characters per line
                per_line = max(35, int(math.ceil(len(q_text) / 3)))
                wrapped = textwrap.fill(q_text, width=per_line, break_long_words=False, break_on_hyphens=False)
                txt_ax.text(0.0, 0.5, wrapped, ha='left', va='center', fontsize=11, linespacing=1.2, wrap=True)

                # Right column: chart
                ax = fig.add_subplot(gs[row_idx, 1])
                groups: List[Tuple[str, Dict[int, float]]] = [('Overall', item['pct'])]
                if sectors:
                    for s in sectors[: max_groups - 1]:
                        sp = item.get('sector_pct', {}).get(s, {})
                        groups.append((str(s), sp))
                _draw_likert_stacked(ax, groups, title="", show_legend=False, title_fs=10)

            # Shared legend at the bottom across the figure
            from matplotlib.patches import Patch  # type: ignore
            colors = _likert_colors()
            handles = [Patch(facecolor=colors[i], edgecolor='white', label=str(i+1)) for i in range(5)]
            # Place legend slightly above the scale caption
            fig.legend(
                handles=handles,
                title='Scale (1–5)',
                loc='lower center',
                bbox_to_anchor=(0.5, 0.10),
                ncol=5,
                frameon=False,
                fontsize=9,
                title_fontsize=9,
            )
            # Add full scale description caption at bottom
            fig.text(0.5, 0.03, scale_desc, ha='center', va='bottom', fontsize=9, color='dimgray')
            # Bottom margin only; spacing handled via GridSpec
            fig.subplots_adjust(bottom=0.22, top=0.95)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Composites overall
        comps = summary.get('composites', [])
        if comps:
            # Overall bar of composite means (vertical to match other charts)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            names = [c['name'] for c in comps]
            means = [c['mean'] if c['mean'] is not None else 0.0 for c in comps]
            x = range(len(names))
            bars = ax.bar(x, means, color='#4C78A8', width=0.5)
            # Shorten names for x labels to keep tidy
            short = []
            for n in names:
                key = n.lower()
                if 'overlap' in key:
                    short.append('Overlap')
                elif 'communication' in key:
                    short.append('Comm')
                elif 'governance' in key:
                    short.append('Gov')
                elif 'ambiguity' in key:
                    short.append('Ambiguity')
                else:
                    short.append(n.replace(' Index',''))
            ax.set_xticks(list(x))
            ax.set_xticklabels(short, rotation=0, ha='center')
            ax.set_ylim(1, 5)
            ax.set_ylabel('Mean (1-5)')
            ax.set_title('Composite Indices (Overall Means)')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            # Annotate values atop bars
            for rect, val in zip(bars, means):
                ax.text(rect.get_x() + rect.get_width()/2, val + 0.03, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # By Role (if available)
            if roles:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                x = range(len(comps))
                width = max(0.1, 0.6 / max(1, len(roles)))
                for i, r in enumerate(roles[:6]):  # cap to 6 roles for readability
                    vals = [c.get('role_means', {}).get(r) for c in comps]
                    vals = [v if v is not None else 0.0 for v in vals]
                    ax.bar([xx + i*width for xx in x], vals, width=width, label=str(r))
                ax.set_xticks([xx + (min(len(roles),6)-1)*width/2 for xx in x])
                ax.set_xticklabels(names, rotation=20, ha='right')
                ax.set_ylim(1, 5)
                ax.set_ylabel('Mean (1-5)')
                ax.set_title('Composite Indices by Role')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # By Sector (if available)
            if sectors:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                x = range(len(comps))
                width = max(0.1, 0.6 / max(1, len(sectors)))
                for i, s in enumerate(sectors[:6]):
                    vals = [c.get('sector_means', {}).get(s) for c in comps]
                    vals = [v if v is not None else 0.0 for v in vals]
                    ax.bar([xx + i*width for xx in x], vals, width=width, label=str(s))
                ax.set_xticks([xx + (min(len(sectors),6)-1)*width/2 for xx in x])
                ax.set_xticklabels(names, rotation=20, ha='right')
                ax.set_ylim(1, 5)
                ax.set_ylabel('Mean (1-5)')
                ax.set_title('Composite Indices by Sector')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Optional: Associations text page
        corrs = summary.get('correlations', [])
        if corrs:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.06, 0.95, 'Associations (Spearman rho)', transform=ax.transAxes, fontsize=16, va='top')
            y = 0.9
            for c in corrs:
                rho = c.get('rho')
                ax.text(0.08, y, f"- {c['name']}: {'n/a' if rho is None else f'{rho:.2f}'}", transform=ax.transAxes, fontsize=12, va='top')
                y -= 0.05
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# -----------------------------
# Minimal PDF Writer (text only)
# -----------------------------

class PDFBuilder:
    def __init__(self, page_width: int = 612, page_height: int = 792, margin: int = 50):
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin
        self.pages: List[bytes] = []  # content streams for each page

    @staticmethod
    def _escape_text(s: str) -> str:
        s = _coerce_latin1(s)
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        s = s.replace("\r", "")
        return s

    def build_from_lines(self, lines: Iterable[Tuple[str, int, int]]):
        """Build pages from (text, font_size, indent_px) lines.
        Performs simple wrapping based on mono-width approximation.
        """
        y = self.page_height - self.margin
        content = []  # type: List[str]

        for text, font_size, indent_px in lines:
            # wrap line approximately by character count
            usable_width = self.page_width - self.margin * 2 - indent_px
            avg_char_px = max(font_size * 0.6, 1.0)
            max_chars = max(int(usable_width / avg_char_px), 8)
            paragraphs = text.split("\n") if text else [""]

            for para in paragraphs:
                # simple word-wrap
                words = para.split()
                if not words:
                    chunk_lines = [""]
                else:
                    chunk_lines = []
                    cur = words[0]
                    for w in words[1:]:
                        if len(cur) + 1 + len(w) <= max_chars:
                            cur += " " + w
                        else:
                            chunk_lines.append(cur)
                            cur = w
                    chunk_lines.append(cur)

                for ln in chunk_lines:
                    if y < self.margin + font_size + 2:
                        # flush current page
                        self.pages.append("".join(content).encode("latin-1", errors="replace"))
                        content = []
                        y = self.page_height - self.margin
                    x = self.margin + indent_px
                    esc = self._escape_text(ln)
                    content.append(f"BT /F1 {font_size} Tf {x} {y} Td ({esc}) Tj ET\n")
                    y -= max(int(font_size * 1.4), 12)

        # flush last page
        if content or not self.pages:
            self.pages.append("".join(content).encode("latin-1", errors="replace"))

    def to_pdf_bytes(self) -> bytes:
        objects: List[bytes] = []
        # 1: Catalog
        # 2: Pages
        # 3: Font (Helvetica)
        # Then: per-page (content, page) pairs

        def obj(n: int, body: str) -> bytes:
            return f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")

        # Font object
        font_obj_num = 3
        objects.append(obj(font_obj_num, "<< /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica >>"))

        # Per-page content and page objects
        page_obj_nums: List[int] = []
        next_obj_num = 4
        for content in self.pages:
            content_num = next_obj_num
            next_obj_num += 1
            page_num = next_obj_num
            next_obj_num += 1

            stream = b"stream\n" + content + b"endstream\n"
            header = f"<< /Length {len(content)} >>\n".encode("latin-1")
            objects.append(obj(content_num, header.decode("latin-1") + stream.decode("latin-1")))

            page_dict = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.page_width} {self.page_height}] "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                f"/Contents {content_num} 0 R >>"
            )
            objects.append(obj(page_num, page_dict))
            page_obj_nums.append(page_num)

        # Pages object (2)
        kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
        pages_body = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>"
        objects.insert(0, obj(2, pages_body))

        # Catalog (1)
        catalog_body = "<< /Type /Catalog /Pages 2 0 R >>"
        objects.insert(0, obj(1, catalog_body))

        # Assemble PDF
        pdf = bytearray()
        pdf.extend(b"%PDF-1.4\n")

        xref: List[int] = [0]  # xref[0] is free object
        for o in objects:
            xref.append(len(pdf))
            pdf.extend(o)

        # xref table
        startxref = len(pdf)
        total_objs = len(objects)
        pdf.extend(f"xref\n0 {total_objs + 1}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for off in xref[1:]:
            pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

        # trailer
        pdf.extend(f"trailer << /Size {total_objs + 1} /Root 1 0 R >>\n".encode("latin-1"))
        pdf.extend(f"startxref\n{startxref}\n%%EOF\n".encode("latin-1"))
        return bytes(pdf)


# -----------------------------
# Report composition
# -----------------------------

def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        if math.isfinite(x):
            return f"{x:.{digits}f}"
        return str(x)
    except Exception:
        return str(x)


def build_report_lines(summary: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    lines: List[Tuple[str, int, int]] = []  # (text, font_size, indent_px)

    def add(text: str, size: int = 11, indent: int = 0):
        lines.append((text, size, indent))

    # Header
    add("CSV Analysis Report", 16, 0)
    add("", 8, 0)
    add(f"File: {os.path.basename(summary['path'])}", 11, 0)
    add(f"Generated: {summary['generated_at']}", 10, 0)
    add("", 8, 0)

    if summary.get("mode") == "likert":
        # Likert-specific layout
        add("Overview", 14, 0)
        add(f"Responses: {summary['rows']}", 11, 15)
        role_col = summary.get('role_col')
        if role_col:
            add(f"Role column: {role_col}", 10, 15)
        roles = summary.get('roles', [])
        if roles:
            add("Roles:", 10, 15)
            for r in roles:
                add(f"- {r}", 10, 25)
        add("Scale: 1=Strongly Disagree … 5=Strongly Agree", 10, 15)
        add("", 8, 0)

        # Per-question summary
        add("Per-Question Summary", 14, 0)
        for item in summary['items']:
            label = item['label']
            label_short = label if len(label) <= 120 else label[:117] + '...'
            add(f"{item['id']}: {label_short}", 11, 10)
            add(
                " | ".join([
                    f"Mean: {_fmt_float(item.get('mean'))}",
                    f"Median: {_fmt_float(item.get('median'))}",
                    f"SD: {_fmt_float(item.get('stdev'))}",
                    f"Missing: {item.get('missing', 0)}",
                ]),
                10,
                20,
            )
            add(
                " | ".join([
                    f"%1:{_fmt_float(item['pct'][1],1)}",
                    f"%2:{_fmt_float(item['pct'][2],1)}",
                    f"%3:{_fmt_float(item['pct'][3],1)}",
                    f"%4:{_fmt_float(item['pct'][4],1)}",
                    f"%5:{_fmt_float(item['pct'][5],1)}",
                    f"Agree:{_fmt_float(item['bands']['agree_pct'],1)}%",
                ]),
                10,
                20,
            )
            if item.get('floor') or item.get('ceiling'):
                add(f"Flags: {'Floor ' if item.get('floor') else ''}{'Ceiling' if item.get('ceiling') else ''}", 10, 20)
            # role means (compact)
            rms = item.get('role_means') or {}
            if rms:
                parts = [f"{r}:{_fmt_float(v,2)}" for r, v in rms.items() if v is not None]
                if parts:
                    add("Roles: " + ", ".join(parts), 10, 20)
            add("", 6, 0)

        # Role deltas
        if summary.get('top_role_deltas'):
            add("Largest Role Differences", 14, 0)
            for d in summary['top_role_deltas']:
                r1, r2 = d['roles']
                m1, m2 = d['means']
                add(f"{d['item']}: {r1} vs {r2} Δ={_fmt_float(d['delta'],2)} ({_fmt_float(m1,2)} vs {_fmt_float(m2,2)})", 10, 15)
            add("", 6, 0)

        # Composites
        if summary.get('composites'):
            add("Composite Indices", 14, 0)
            for c in summary['composites']:
                items = ", ".join(c['items'])
                rc = c.get('reverse') or []
                rc_str = f" (rev: {', '.join(rc)})" if rc else ""
                add(f"{c['name']}: mean={_fmt_float(c['mean'],2)} SD={_fmt_float(c['stdev'],2)} α={_fmt_float(c.get('alpha'),2)}", 10, 10)
                add(f"Items: {items}{rc_str}", 9, 20)
                rms = c.get('role_means') or {}
                parts = [f"{r}:{_fmt_float(v,2)}" for r, v in rms.items() if v is not None]
                if parts:
                    add("Roles: " + ", ".join(parts), 9, 20)
                add("", 4, 0)

        # Governance vs Ambiguity gap
        if summary.get('gap'):
            gap = summary['gap']
            add("Governance vs Ambiguity", 14, 0)
            add(f"Governance agree: {_fmt_float(gap['gov_agree_pct'],1)}% | Ambiguity agree: {_fmt_float(gap['amb_agree_pct'],1)}%", 10, 10)
            q = gap.get('quadrants', {})
            add(f"Quadrants (Gov/ Amb): HH={q.get('HH',0)} HL={q.get('HL',0)} LH={q.get('LH',0)} LL={q.get('LL',0)}", 10, 10)
            add("", 6, 0)

        # Associations
        if summary.get('correlations'):
            add("Associations (Spearman ρ)", 14, 0)
            for c in summary['correlations']:
                rho = c.get('rho')
                add(f"{c['name']}: {('n/a' if rho is None else f'{rho:.2f}')}", 10, 10)
            add("", 6, 0)

        return lines

    # Generic CSV layout (fallback)
    add("Overview", 14, 0)
    add(f"Rows: {summary['rows']}", 11, 15)
    add(f"Columns: {summary['cols']}", 11, 15)
    add(f"Missing cells: {summary['missing_cells']} ({_fmt_float(summary['missing_pct'], 2)}%)", 11, 15)
    add("", 8, 0)

    add("Columns", 14, 0)
    for col in summary["columns"]:
        add(f"{col['name']}", 12, 10)
        add(f"Type: {col['type']}", 10, 20)
        add(f"Missing: {col['missing']} of {col['missing'] + col['non_missing']} ({_fmt_float(col['missing_pct'], 2)}%)", 10, 20)
        if col["type"] == "numeric":
            add(" | ".join([
                f"Count: {col.get('count', 0)}",
                f"Mean: {_fmt_float(col.get('mean'))}",
                f"Median: {_fmt_float(col.get('median'))}",
                f"Std: {_fmt_float(col.get('stdev'))}",
            ]), 10, 20)
            add(" | ".join([
                f"Min: {_fmt_float(col.get('min'))}",
                f"Max: {_fmt_float(col.get('max'))}",
            ]), 10, 20)
        elif col["type"] == "categorical":
            add(f"Unique values: {col.get('unique', 0)}", 10, 20)
            top = col.get("top", [])
            if top:
                for val, cnt in top:
                    safe_val = (val if len(str(val)) <= 80 else str(val)[:77] + "...")
                    add(f"- {safe_val} — {cnt}", 10, 28)
        add("", 6, 0)

    return lines


def write_pdf_report(summary: Dict[str, Any], output_path: str) -> None:
    if HAS_MPL:
        try:
            export_prefix = os.path.splitext(output_path)[0]
            render_pdf_with_matplotlib(summary, output_path, export_prefix=export_prefix)
            return
        except Exception as e:
            # Surface plotting error and fall back to text-only
            print(f"[warn] Matplotlib rendering failed: {e.__class__.__name__}: {e}")
            # Uncomment next line to debug full trace locally
            # import traceback; traceback.print_exc()
    builder = PDFBuilder()
    lines = build_report_lines(summary)
    builder.build_from_lines(lines)
    data = builder.to_pdf_bytes()
    with open(output_path, "wb") as f:
        f.write(data)


def main(argv: List[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python analyze_csv_to_pdf.py input.csv [output.pdf]")
        return 2
    input_path = argv[1]
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return 1
    if len(argv) == 3:
        output_path = argv[2]
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = base + "_summary.pdf"

    # Try Likert analysis first; fallback to generic
    likert = analyze_likert_csv(input_path)
    summary = likert if likert is not None else analyze_csv(input_path)
    write_pdf_report(summary, output_path)
    print(f"Wrote PDF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
