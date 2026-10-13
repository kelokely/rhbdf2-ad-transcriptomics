#!/usr/bin/env python3
"""Run gated donor-aware ligand-receptor node-overlay analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.ligand_receptor_utils import (
    build_donor_aware_lr_overlay,
    fetch_omnipath_ligand_receptor,
    plot_lr_replication,
    summarize_lr_replication,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", default="results/independent_ad_cohorts/synthesis")
    parser.add_argument("--output", default="results/independent_ad_cohorts/ligand_receptor")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    synthesis = _resolve(root, args.synthesis)
    output = _resolve(root, args.output)
    gate = pd.read_csv(synthesis / "tables/ligand_receptor_gate.csv")
    if gate.iloc[0]["status"] not in {"eligible_not_run", "completed"}:
        raise RuntimeError(f"Ligand-receptor gate is not open: {gate.iloc[0]['status']}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    interactions = fetch_omnipath_ligand_receptor(tables / "omnipath_ligrecextra_raw.tsv")
    interactions.to_csv(tables / "omnipath_ligrecextra_curated.csv", index=False)
    cohort_dirs = {"GSE243292": root / "results/GSE243292_multicell_AD_extension"}
    independent_root = root / "results/independent_ad_cohorts"
    for cohort_dir in sorted(path for path in independent_root.iterdir() if path.is_dir()):
        if (cohort_dir / "cell_types").exists():
            cohort_dirs[cohort_dir.name] = cohort_dir
    edges, status = build_donor_aware_lr_overlay(cohort_dirs, interactions)
    replication = summarize_lr_replication(edges)
    edges.to_csv(tables / "donor_fdr_supported_lr_edges_all_cohorts.csv", index=False)
    replication.to_csv(tables / "donor_fdr_supported_lr_replication.csv", index=False)
    status.to_csv(tables / "lr_input_status.csv", index=False)
    plot_lr_replication(replication, figures / "replicated_ligand_receptor_edges.png")
    replicated = replication[replication["replicating_cohort_count"].ge(2)]
    lines = [
        "# Donor-aware ligand-receptor node overlay",
        "",
        f"- OmniPath ligrecextra directed records: {interactions.shape[0]:,}",
        f"- Within-cohort edges with donor-FDR support at both nodes: {edges.shape[0]:,}",
        f"- Directed cell-pair edges reproduced in at least two cohorts: {replicated.shape[0]:,}",
        "",
        "This layer overlays donor-level RHBDF2 association statistics on curated ligand-receptor edges. It is not a direct assay of secretion, receptor occupancy, protein activity, physical proximity, or signaling flux. Marker-derived cell labels and transcript abundance remain limitations.",
    ]
    (output / "ligand_receptor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Donor-aware ligand-receptor overlay completed: {output}")
    return 0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
