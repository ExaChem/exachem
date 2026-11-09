#!/usr/bin/env python3
"""
Analyze QFlow calculation output and visualize convergence.

Example: -i cycle1.out cycle2.out -n 56860 --image c3h8.png --target -118.774623625914145

Cycles are read from the "[Cycle N]" tags, so the input may be one file with
all cycles, one file per cycle, or any mix. If a cycle appears in more than one
file (e.g. a partial cycle rerun after a restart), the copy in the file listed
last replaces the earlier ones in full.

Parse the exachem output file for a qflow calculation containing repeated blocks like:

  [Cycle 1] Combination (10657/56860): 10 11 12 ...
  Final Energy: -118.747972

Outputs:
- CSV with columns:
    cycle, pos_in_output, combination, energy
  where:
    cycle = 1-based iteration index
    pos_in_output = the position within that cycle as it appears in the output file
    combination = the actual Combination in (1...ncombinations) that is processed" 
    energy = parsed final energy

- plot (png)

Requirements:
  pip install plotly kaleido
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

COMBO_RE = re.compile(r"^\s*\[\s*Cycle\s+(\d+)\s*\]\s*Combination\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*:\s*(.*)\s*$")
ENERGY_RE = re.compile(
    r"^\s*Final\s+Energy\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


Record = Tuple[int, int, int, float]  # (iter, pos_in_output, combination, energy)


def parse_file(path: Path, expected_end: int) -> Tuple[Dict[int, List[Record]], List[str]]:
    """
    Parse one output file into records grouped by cycle.

    The cycle is taken from the "[Cycle N]" tag; pos is the order in which
    that cycle's energies appear in the file.
    """
    warnings: List[str] = []
    by_cycle: Dict[int, List[Record]] = {}

    cycle: Optional[int] = None
    pending_x: Optional[int] = None
    pending_total: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            m = COMBO_RE.match(line)
            if m:
                c = int(m.group(1))
                x = int(m.group(2))
                total = int(m.group(3))

                cycle = c

                if total != expected_end:
                    warnings.append(
                        f"{path}:{lineno}: Found total={total} but expected {expected_end}. Continuing anyway."
                    )

                pending_x = x
                pending_total = total
                continue

            m = ENERGY_RE.match(line)
            if m:
                if pending_x is None:
                    warnings.append(
                        f"{path}:{lineno}: Found 'Final Energy' without a preceding 'Combination' line. Ignoring."
                    )
                    continue

                energy = float(m.group(1))

                recs = by_cycle.setdefault(cycle, [])
                recs.append((cycle, len(recs) + 1, pending_x, energy))

                pending_x = None
                pending_total = None
                continue

    if pending_x is not None:
        warnings.append(
            f"{path}: EOF reached after 'Combination (x/{pending_total})' for x={pending_x}, but no following 'Final Energy'."
        )

    return by_cycle, warnings


def write_csv(records: List[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "pos_in_output", "combination", "energy"])
        w.writerows(records)


def write_plot_image(
    records: List[Record],
    expected_end: int,
    out_image: Path,
    target: Optional[float],
    width: int,
    height: int,
    scale: float,
) -> None:
    import plotly.graph_objects as go

    # --- helpers ---
    def adjust_tick_positions(values_true, span, fig_height_px, tick_font_size=14) -> list[float]:
        """
        Plotly won't avoid overlaps for tickmode="array". We keep the *text* as the true values,
        but slightly nudge the *positions* so labels don't overlap.
        """
        if not values_true:
            return []

        usable_h = max(200, int(fig_height_px * 0.70))
        desired_px = tick_font_size + 6
        min_sep_data = (span / usable_h) * desired_px

        vals = sorted(values_true)
        adjusted = [vals[0]]
        for v in vals[1:]:
            prev = adjusted[-1]
            if v - prev < min_sep_data:
                adjusted.append(prev + min_sep_data)
            else:
                adjusted.append(v)
        return adjusted

    # --- Group by cycle ---
    by_cycle: dict[int, list[tuple[int, float]]] = {}
    energies: list[float] = []

    # records: (cycle, pos_in_output, combination, energy)
    for cycle, pos, _x, e in records:
        by_cycle.setdefault(cycle, []).append((pos, e))
        energies.append(e)

    fig = go.Figure()

    start_energies: list[float] = []
    end_energies_complete: list[float] = []  # right-axis ticks for completed cycles only

    # Plot each cycle + handle incomplete-cycle combined annotation
    for cycle in sorted(by_cycle.keys()):
        pts = sorted(by_cycle[cycle], key=lambda t: t[0])  # (pos, energy)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                name=f"Cycle {cycle}",
                line=dict(width=1),
            )
        )

        start_pos, start_e = pts[0]
        last_pos, last_e = pts[-1]
        start_energies.append(start_e)

        # Incomplete cycle: ends at least 100 before expected_end
        incomplete = (expected_end - last_pos) >= 100

        if incomplete:
            # Annotate combined label near endpoint: "Cycle X@pos: energy"
            fig.add_annotation(
                x=last_pos,
                y=last_e,
                text=f"Cycle {cycle}@{last_pos}: {last_e:.4f}",
                showarrow=False,
                xanchor="center",
                yanchor="top",          # anchor the text box from its top
                yshift=-10,             # shift it downward (pixels) so it sits below the line
                font=dict(size=14),
                bgcolor="white",
            )
        else:
            # Completed (or close enough): show end energy as a y2 tick
            end_energies_complete.append(last_e)

    # Compute y-range and include target
    ymin = min(energies) if energies else 0.0
    ymax = max(energies) if energies else 1.0
    if target is not None:
        ymin = min(ymin, target)
        ymax = max(ymax, target)

    span = (ymax - ymin) if ymax > ymin else 1.0
    pad = 0.03 * span
    y_range = [ymin - pad, ymax + pad]
    span_padded = y_range[1] - y_range[0]

    # Target reference line
    if target is not None:
        fig.add_hline(y=target, line_width=3, line_dash="dash")

    # --- LEFT axis ticks: cycle starts + ref ---
    left_vals_true = sorted(set(start_energies + ([target] if target is not None else [])))
    y_tickvals = left_vals_true
    y_ticktext = [
        f"ref: {v:.4f}" if (target is not None and abs(v - target) < 1e-12)
        else f"{v:.4f}"
        for v in left_vals_true
    ]


    # --- RIGHT axis ticks: cycle ends (completed only) + ref ---
    right_vals_true = sorted(set(end_energies_complete + ([target] if target is not None else [])))

    # Adjust right tick positions to prevent overlap (text stays the true value)
    y2_tickvals = adjust_tick_positions(
        values_true=right_vals_true,
        span=span_padded,
        fig_height_px=height,
        tick_font_size=14,
    )
    y2_ticktext = [f"{v:.4f}" for v in right_vals_true]

    # Force yaxis2 to render in static export by adding an invisible trace bound to y2
    if y2_tickvals:
        fig.add_trace(
            go.Scatter(
                x=[1, 1],
                y=[y_range[0], y_range[1]],
                yaxis="y2",
                mode="markers",
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    mol_name = Path(out_image).stem
    fig.update_layout(
        title=dict(
            text=f"{mol_name} - Final Energy vs Combination (per Cycle)",
            font=dict(size=22),
        ),
        xaxis=dict(
            title=dict(
                text=f"Index within cycle (1..{expected_end})",
                font=dict(size=18),
            ),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="Final Energy (Cycle starts)", font=dict(size=18)),
            range=y_range,
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            tickfont=dict(size=14),
            showgrid=True,
            showline=True,
            ticks="outside",
        ),
        yaxis2=dict(
            title=dict(text="Final Energy (Cycle ends)", font=dict(size=18)),
            overlaying="y",
            side="right",
            range=y_range,
            tickmode="array",
            tickvals=y2_tickvals,
            ticktext=[f" {t}" for t in y2_ticktext],  # leading space helps minus sign vs axis line
            tickfont=dict(size=14),
            showgrid=False,
            showticklabels=True,
            showline=True,
            linecolor="black",
            linewidth=1.5,
            ticks="outside",
            ticklen=8,
            tickwidth=1.5,
            anchor="x",
            position=1.0,
        ),
        legend=dict(
            title=dict(text="Cycle", font=dict(size=16)),
            font=dict(size=14),
        ),
        margin=dict(l=140, r=170, t=80, b=80),
    )

    out_image.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_image), width=width, height=height, scale=scale)
    #html_fn = Path(out_image).stem + ".html"
    #fig.write_html(html_fn, include_plotlyjs=True)
    print(f"Wrote plot image to {out_image}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract per-cycle energies from QFlow output files and save a static Plotly image."
    )
    ap.add_argument("-i", "--input", nargs="+", required=True, help="Output file(s) in job order; a cycle in a later file replaces earlier copies.")
    ap.add_argument(
        "-n", "--end", type=int, required=True, help="Total number of combinations per cycle (e.g., 56860)."
    )
    ap.add_argument("--image", required=True, help="Output image file (e.g., plot.png, plot.pdf, plot.svg).")
    ap.add_argument("--csv", type=bool, default=True, help="Output CSV path (default: image-name.csv).")
    ap.add_argument("--target", type=float, default=None, help="Optional target energy (dashed horizontal line).")

    ap.add_argument("--width", type=int, default=2000, help="Image width in pixels (default: 2000).")
    ap.add_argument("--height", type=int, default=1100, help="Image height in pixels (default: 1100).")
    ap.add_argument("--scale", type=float, default=2.0, help="Scale factor for high DPI (default: 2.0).")

    ap.add_argument("--quiet", action="store_true", help="Suppress warnings.")
    ap.add_argument(
        "--check-x",
        action="store_true",
        help=(
            "Extra sanity check: within each iteration, report duplicates/missing combination values "
            "(does not affect chunking)."
        ),
    )
    args = ap.parse_args()

    # A cycle found in a later file replaces it in full (restarts rerun a cycle from scratch).
    cycles: Dict[int, List[Record]] = {}
    source: Dict[int, Path] = {}
    warnings: List[str] = []
    for name in args.input:
        in_path = Path(name)
        if not in_path.exists():
            print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
            return 2
        by_cycle, warns = parse_file(in_path, args.end)
        warnings.extend(warns)
        if not by_cycle:
            warnings.append(f"{in_path}: no (Combination, Final Energy) pairs found.")
        for cycle, recs in by_cycle.items():
            if cycle in cycles:
                warnings.append(
                    f"cycle {cycle}: replacing {len(cycles[cycle])} energies from {source[cycle]} "
                    f"with {len(recs)} from {in_path}."
                )
            cycles[cycle] = recs
            source[cycle] = in_path

    records: List[Record] = []
    for cycle in sorted(cycles):
        n = len(cycles[cycle])
        if n < args.end:
            warnings.append(f"cycle {cycle} ({source[cycle]}) has {n}/{args.end} energies (incomplete).")
        records.extend(cycles[cycle])

    if not records:
        print("WARNING: No (Combination, Final Energy) pairs found.", file=sys.stderr)
        return 1

    if args.csv:
        csv_fn = Path(args.image).stem + ".csv"
        write_csv(records, Path(csv_fn))
        print(f"Wrote {len(records)} records to {csv_fn}")

    counts: dict[int, int] = {}
    for it, _, _, _ in records:
        counts[it] = counts.get(it, 0) + 1

    print(f"Detected {len(counts)} cycle(s) (END={args.end})")
    for it in sorted(counts):
        print(f"  cycle {it}: {counts[it]} energies")

    if args.check_x:
        # Optional sanity: duplicates/missing combination within each iteration
        # (Useful if you expect each iter to cover all x=1..END exactly once.)
        by_iter_x: dict[int, List[int]] = {}
        for it, _pos, x, _e in records:
            by_iter_x.setdefault(it, []).append(x)

        for it in sorted(counts):
            xs = by_iter_x.get(it, [])
            if not xs:
                continue
            uniq = set(xs)
            dup_count = len(xs) - len(uniq)
            missing = args.end - len(uniq) if args.end > 0 else 0
            # Note: "missing" assumes x values should be 1..END; if not, ignore.
            print(f"  [check-x] cycle {it}: unique x={len(uniq)}, duplicates={dup_count}, approx-missing={missing}")

    if warnings and not args.quiet:
        print("\nWarnings:", file=sys.stderr)
        for w in warnings[:50]:
            print(" - " + w, file=sys.stderr)
        if len(warnings) > 50:
            print(f" - ... plus {len(warnings) - 50} more", file=sys.stderr)

    write_plot_image(
        records=records,
        expected_end=args.end,
        out_image=Path(args.image),
        target=args.target,
        width=args.width,
        height=args.height,
        scale=args.scale,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

