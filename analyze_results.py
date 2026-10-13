#!/usr/bin/env python3
"""Summarize completed pipeline runs into compact markdown analysis reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed pipeline runs and write compact markdown reports."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        required=True,
        help="Path to a completed results/<run_name> directory. Provide this flag multiple times.",
    )
    parser.add_argument(
        "--cross-dataset-dir",
        default="results/cross_dataset_validation",
        help="Optional directory containing cross-dataset validation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="report/generated_reports",
        help="Directory where the generated markdown reports should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_reports: list[dict[str, str]] = []
    for run_dir_value in args.run_dirs:
        run_dir = Path(run_dir_value).expanduser().resolve()
        report_path = write_dataset_report(run_dir=run_dir, output_dir=output_dir)
        dataset_reports.append({"label": run_dir.name, "path": str(report_path)})

    cross_dataset_dir = Path(args.cross_dataset_dir).expanduser().resolve()
    cross_report_path = None
    if cross_dataset_dir.exists():
        cross_report_path = write_cross_dataset_report(
            cross_dataset_dir=cross_dataset_dir,
            output_dir=output_dir,
        )

    index_path = write_report_index(
        output_dir=output_dir,
        dataset_reports=dataset_reports,
        cross_report_path=cross_report_path,
    )
    print(f"Analysis reports written to: {output_dir}")
    print(f"Index: {index_path}")
    return 0


def write_dataset_report(run_dir: Path, output_dir: Path) -> Path:
    config = _load_yaml(run_dir / "summaries" / "config_used.yaml")
    dataset_name = str(config.get("dataset_name", run_dir.name))
    state_summary = pd.read_csv(run_dir / "summaries" / "state_summary.csv")
    metadata_summary = _load_optional_csv(run_dir / "tables" / "state_metadata_association_summary.csv")
    candidate_summary = pd.read_csv(run_dir / "tables" / "candidate_evidence_summary.csv")
    module_status = pd.read_csv(run_dir / "summaries" / "module_status_summary.csv")
    plot_status = pd.read_csv(run_dir / "summaries" / "plot_status_summary.csv")
    top_positive = _load_optional_csv(run_dir / "tables" / "top_positive_genes.csv")
    pseudobulk_positive = _load_optional_csv(run_dir / "pseudobulk" / "pseudobulk_top_positive_genes.csv")
    continuous_positive = _load_optional_csv(run_dir / "continuous" / "continuous_top_positive_genes.csv")
    protein_context = _load_optional_csv(run_dir / "external_evidence" / "protein_context_evidence_summary.csv")
    public_proteomics = _load_optional_csv(run_dir / "external_evidence" / "public_proteomics_overlap_summary.csv")
    activity_proxy = _load_optional_csv(run_dir / "activity_proxies" / "pathway_activity_proxy_summary.csv")
    subpopulation_run_summary = _load_optional_csv(
        run_dir / "pseudobulk" / "subpopulation_pseudobulk" / "subpopulation_pseudobulk_run_summary.csv"
    )
    subpopulation_gene_summary = _load_optional_csv(
        run_dir / "pseudobulk" / "subpopulation_pseudobulk" / "subpopulation_pseudobulk_gene_summary.csv"
    )

    overview = _parse_dataset_summary(run_dir / "summaries" / "dataset_summary.txt")
    high_fraction = _derive_high_fraction(state_summary)
    candidate_table = _summarize_candidate_support(candidate_summary)
    metadata_table = _summarize_metadata_associations(metadata_summary)
    module_lines = _build_status_lines(module_status)
    plot_lines = _build_status_lines(plot_status, only_problematic=True)
    protein_context_lines = _summarize_protein_context(protein_context)
    public_proteomics_lines = _summarize_public_proteomics(public_proteomics)
    activity_proxy_table = _summarize_activity_proxy(activity_proxy)
    subpopulation_lines = _summarize_subpopulation_pseudobulk(
        run_summary=subpopulation_run_summary,
        gene_summary=subpopulation_gene_summary,
    )

    lines = [
        f"# Analysis Report: {dataset_name}",
        "",
        "## Overview",
        f"- Run directory: `{run_dir}`",
        f"- Input path: `{config.get('input_path', 'n/a')}`",
        f"- Cells: `{overview.get('Cells', 'n/a')}`",
        f"- Genes: `{overview.get('Genes', 'n/a')}`",
        f"- Gene of interest: `{config.get('gene_of_interest', 'RHBDF2')}`",
        f"- `RHBDF2_high` fraction: `{high_fraction:.3f}`" if pd.notna(high_fraction) else "- `RHBDF2_high` fraction: `n/a`",
        "",
        "## State Counts",
        *_state_count_lines(state_summary),
        "",
        "## Candidate Evidence Summary",
        _markdown_table(candidate_table),
        "",
        "## Top Gene Lists",
        f"- Cell-level positive genes: {_gene_preview(top_positive)}",
        f"- Pseudobulk positive genes: {_gene_preview(pseudobulk_positive)}",
        f"- Continuous positive genes: {_gene_preview(continuous_positive)}",
        "",
        "## Protein Context",
        *protein_context_lines,
        "",
        "## Public Proteomics Overlap",
        *public_proteomics_lines,
        "",
        "## Pathway Activity Proxies",
        _markdown_table(activity_proxy_table),
        "",
        "## Within-Subpopulation Pseudobulk",
        *subpopulation_lines,
        "",
        "## Strongest Metadata Associations",
        _markdown_table(metadata_table),
        "",
        "## Module Status",
        *module_lines,
        "",
        "## Plot Status",
        *(plot_lines or ["- All recorded plots completed successfully."]),
        "",
        "## Existing Run Report",
        f"- `{run_dir / 'run_summary.md'}`",
        "",
    ]

    output_path = output_dir / f"{dataset_name}_analysis_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_cross_dataset_report(cross_dataset_dir: Path, output_dir: Path) -> Path:
    replication = pd.read_csv(cross_dataset_dir / "cross_dataset_candidate_replication.csv")
    run_summary = _load_optional_csv(cross_dataset_dir / "cross_dataset_run_summary.csv")

    replication = replication.copy()
    replication["replication_score"] = (
        replication.get("n_pseudobulk_significant_datasets", 0).fillna(0) * 5
        + replication.get("n_continuous_significant_datasets", 0).fillna(0) * 4
        + replication.get("n_detection_significant_datasets", 0).fillna(0) * 3
        + replication.get("n_positive_expression_significant_datasets", 0).fillna(0) * 2
        + replication.get("n_cell_level_significant_datasets", 0).fillna(0)
        + replication.get("n_subpopulation_supported_datasets", 0).fillna(0) * 2
        + replication.get("n_protein_context_supported_datasets", 0).fillna(0)
        + replication.get("n_public_proteomics_overlap_supported_datasets", 0).fillna(0)
        + replication.get("n_genetics_supported_datasets", 0).fillna(0)
        + replication.get("n_brain_context_supported_datasets", 0).fillna(0)
        + replication.get("mean_weighted_total_evidence_score", 0).fillna(0)
    )
    top_replication = replication.sort_values(
        [
            "replication_score",
            "n_pseudobulk_significant_datasets",
            "n_continuous_significant_datasets",
            "mean_weighted_total_evidence_score",
            "gene",
        ],
        ascending=[False, False, False, False, True],
    ).head(10)

    display = top_replication[
        [
            "gene",
            "replication_score",
            "n_cell_level_significant_datasets",
            "n_pseudobulk_significant_datasets",
            "n_continuous_significant_datasets",
            "n_detection_significant_datasets",
            "n_positive_expression_significant_datasets",
            "n_public_proteomics_overlap_supported_datasets",
            "mean_weighted_total_evidence_score",
        ]
    ].copy()

    lines = [
        "# Cross-Dataset Analysis Report",
        "",
        "## Run Coverage",
    ]
    if run_summary is not None and not run_summary.empty:
        lines.append(_markdown_table(run_summary))
    else:
        lines.append("- Cross-dataset run summary was not available.")

    lines.extend(
        [
            "",
            "## Top Replicated Candidates",
            _markdown_table(display),
            "",
            "## Strongest Replication Calls",
            f"- Genes replicated at pseudobulk level in all available datasets: {_comma_list(_genes_with_threshold(replication, 'n_pseudobulk_significant_datasets', 2))}",
            f"- Genes replicated in continuous modeling in all available datasets: {_comma_list(_genes_with_threshold(replication, 'n_continuous_significant_datasets', 2))}",
            f"- Genes with detection-level support in all available datasets: {_comma_list(_genes_with_threshold(replication, 'n_detection_significant_datasets', 2))}",
            f"- Genes with any subpopulation support: {_comma_list(_genes_with_threshold(replication, 'n_subpopulation_supported_datasets', 1))}",
            f"- Genes with public-proteomics overlap support in all available datasets: {_comma_list(_genes_with_threshold(replication, 'n_public_proteomics_overlap_supported_datasets', 2))}",
            f"- Genes with genetics support in all available datasets: {_comma_list(_genes_with_threshold(replication, 'n_genetics_supported_datasets', 2))}",
            "",
            "## Source File",
            f"- `{cross_dataset_dir / 'cross_dataset_candidate_replication.csv'}`",
            "",
        ]
    )

    output_path = output_dir / "cross_dataset_analysis_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_report_index(
    output_dir: Path,
    dataset_reports: list[dict[str, str]],
    cross_report_path: Path | None,
) -> Path:
    lines = [
        "# Generated Report Index",
        "",
        "## Dataset Reports",
    ]
    for item in dataset_reports:
        lines.append(f"- `{item['label']}`: `{item['path']}`")

    lines.extend(["", "## Cross-Dataset Report"])
    if cross_report_path is None:
        lines.append("- Cross-dataset report was not generated.")
    else:
        lines.append(f"- `{cross_report_path}`")

    index_path = output_dir / "REPORT_INDEX.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def _summarize_candidate_support(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    summary = candidate_summary.copy()
    summary["cell_level_significant"] = summary.get("cell_level_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05
    summary["pseudobulk_significant"] = summary.get("pseudobulk_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05
    summary["continuous_significant"] = summary.get("continuous_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05
    summary["detection_significant"] = summary.get("detection_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05
    summary["positive_expression_significant"] = (
        summary.get("positive_expression_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05
    )
    if "subpopulation_pseudobulk_n_subgroups_significant" in summary.columns:
        summary["subpopulation_supported"] = (
            summary["subpopulation_pseudobulk_n_subgroups_significant"].fillna(0).astype(float) > 0
        )
    else:
        summary["subpopulation_supported"] = False

    support_cols = [
        "cell_level_significant",
        "pseudobulk_significant",
        "continuous_significant",
        "detection_significant",
        "positive_expression_significant",
        "subpopulation_supported",
        "protein_context_supported",
        "public_proteomics_overlap_supported",
        "genetics_supported",
        "brain_context_supported",
    ]
    for flag in ["protein_context_supported", "public_proteomics_overlap_supported", "genetics_supported", "brain_context_supported"]:
        if flag not in summary.columns:
            summary[flag] = False
    summary["support_score"] = summary[support_cols].fillna(False).astype(int).sum(axis=1)
    if "weighted_total_evidence_score" not in summary.columns:
        summary["weighted_total_evidence_score"] = summary["support_score"].astype(float)
    summary = summary.sort_values(
        ["weighted_total_evidence_score", "support_score", "pseudobulk_significant", "cell_level_significant", "gene"],
        ascending=[False, False, False, False, True],
    )

    display = summary[
        [
            "gene",
            "support_score",
            "weighted_total_evidence_score",
            "cell_level_logfoldchange",
            "cell_level_padj",
            "continuous_slope_per_log1p_rhbdf2",
            "continuous_padj",
            "pseudobulk_effect_log1p_cpm",
            "pseudobulk_padj",
            "detection_padj",
            "positive_expression_padj",
            "public_proteomics_overlap_supported",
            "weighted_context_evidence_score",
        ]
    ].copy()
    display = display.head(10)
    for column in display.columns:
        if column == "gene":
            continue
        display[column] = display[column].map(_format_number)
    return display


def _summarize_metadata_associations(metadata_summary: pd.DataFrame | None) -> pd.DataFrame:
    if metadata_summary is None or metadata_summary.empty:
        return pd.DataFrame([{"metadata_column": "n/a", "chi_square_p_value": "n/a", "cramers_v": "n/a"}])
    display = metadata_summary.sort_values("chi_square_p_value", ascending=True, na_position="last").head(5).copy()
    for column in ["chi_square_p_value", "cramers_v"]:
        if column in display.columns:
            display[column] = display[column].map(_format_number)
    return display


def _summarize_protein_context(protein_context: pd.DataFrame | None) -> list[str]:
    if protein_context is None or protein_context.empty:
        return ["- Protein-context summary was not available."]
    supported = protein_context["protein_context_evidence_supported"].fillna(False).astype(bool) if "protein_context_evidence_supported" in protein_context.columns else pd.Series(False, index=protein_context.index)
    overlap = protein_context["public_proteomics_overlap_supported"].fillna(False).astype(bool) if "public_proteomics_overlap_supported" in protein_context.columns else pd.Series(False, index=protein_context.index)
    top = protein_context.loc[supported, "gene"].astype(str).head(8).tolist() if "gene" in protein_context.columns else []
    return [
        f"- Candidate genes with merged protein-context support: `{int(supported.sum())}/{int(len(protein_context))}`",
        f"- Candidate genes with public proteomics overlap support inside the protein-context table: `{int(overlap.sum())}`",
        f"- Top protein-context-supported genes: {_comma_list(top)}",
    ]


def _summarize_public_proteomics(public_proteomics: pd.DataFrame | None) -> list[str]:
    if public_proteomics is None or public_proteomics.empty:
        return ["- Public proteomics overlap summary was not available."]
    detected = public_proteomics["proteomics_gene_detected"].fillna(False).astype(bool).sum() if "proteomics_gene_detected" in public_proteomics.columns else 0
    supported = public_proteomics["public_proteomics_overlap_supported"].fillna(False).astype(bool) if "public_proteomics_overlap_supported" in public_proteomics.columns else pd.Series(False, index=public_proteomics.index)
    top = public_proteomics.loc[supported, "gene"].astype(str).head(8).tolist() if "gene" in public_proteomics.columns else []
    proteomics_context_label = "public proteomics dataset"
    if "proteomics_context_label" in public_proteomics.columns:
        non_empty_labels = (
            public_proteomics["proteomics_context_label"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        non_empty_labels = non_empty_labels[non_empty_labels != ""]
        if not non_empty_labels.empty:
            proteomics_context_label = non_empty_labels.iloc[0]
    return [
        f"- Candidate genes detected in the {proteomics_context_label}: `{int(detected)}`",
        f"- Candidate genes with directionally concordant transcript and public proteomics support: `{int(supported.sum())}`",
        f"- Top concordant genes: {_comma_list(top)}",
    ]


def _summarize_activity_proxy(activity_proxy: pd.DataFrame | None) -> pd.DataFrame:
    if activity_proxy is None or activity_proxy.empty:
        return pd.DataFrame([{"module_name": "n/a", "cell_level_padj": "n/a", "donor_paired_padj": "n/a", "delta": "n/a"}])
    display = activity_proxy[
        [
            column
            for column in [
                "module_name",
                "cell_level_delta_high_minus_low",
                "cell_level_padj",
                "pearson_r_with_RHBDF2",
                "donor_paired_mean_delta",
                "donor_paired_padj",
            ]
            if column in activity_proxy.columns
        ]
    ].copy()
    rename_map = {
        "cell_level_delta_high_minus_low": "delta",
        "pearson_r_with_RHBDF2": "pearson_r",
        "donor_paired_mean_delta": "donor_delta",
    }
    display = display.rename(columns=rename_map)
    for column in display.columns:
        if column == "module_name":
            continue
        display[column] = display[column].map(_format_number)
    return display.head(10)


def _summarize_subpopulation_pseudobulk(
    run_summary: pd.DataFrame | None,
    gene_summary: pd.DataFrame | None,
) -> list[str]:
    if run_summary is None or run_summary.empty:
        return ["- Within-subpopulation pseudobulk was not available."]

    completed = run_summary[run_summary.get("status", "").astype(str) == "completed"].copy()
    lines = [
        f"- Subgroups completed: `{int(completed.shape[0])}`",
        f"- Subgroups attempted: `{int(run_summary.shape[0])}`",
    ]
    if "subgroup_value" in completed.columns and not completed.empty:
        subgroup_preview = [
            f"{row.subgroup_value} (samples={int(row.n_pseudobulk_samples)}, paired_donors={int(row.n_paired_donors)})"
            for row in completed.itertuples(index=False)
        ]
        lines.append(f"- Completed subgroup runs: {_comma_list(subgroup_preview)}")

    if gene_summary is None or gene_summary.empty:
        lines.append("- Gene-level within-subpopulation support was not available.")
        return lines

    sortable = gene_summary.sort_values(
        [
            "subpopulation_pseudobulk_n_subgroups_significant",
            "subpopulation_pseudobulk_n_subgroups_positive",
            "gene",
        ],
        ascending=[False, False, True],
    )
    top_genes = [
        (
            f"{row.gene} "
            f"(significant_subgroups={int(row.subpopulation_pseudobulk_n_subgroups_significant)}, "
            f"mean_effect={_format_number(row.subpopulation_pseudobulk_mean_effect_log1p_cpm)}, "
            f"best={row.subpopulation_pseudobulk_best_subgroup})"
        )
        for row in sortable.head(8).itertuples(index=False)
    ]
    lines.append(f"- Top within-subpopulation genes: {_comma_list(top_genes)}")
    return lines


def _state_count_lines(state_summary: pd.DataFrame) -> list[str]:
    state_column = next((column for column in state_summary.columns if column == "RHBDF2_state"), state_summary.columns[0])
    return [f"- `{row[state_column]}`: `{int(row['count'])}`" for _, row in state_summary.iterrows()]


def _derive_high_fraction(state_summary: pd.DataFrame) -> float:
    if state_summary.empty:
        return float("nan")
    state_column = next((column for column in state_summary.columns if column != "count"), state_summary.columns[0])
    total = float(state_summary["count"].sum())
    high = float(
        state_summary.loc[state_summary[state_column].astype(str).str.contains("high", case=False), "count"].sum()
    )
    return high / total if total else float("nan")


def _build_status_lines(frame: pd.DataFrame, only_problematic: bool = False) -> list[str]:
    if frame.empty:
        return ["- No status rows were recorded."]
    rows = frame.copy()
    if only_problematic and "status" in rows.columns:
        rows = rows[rows["status"].astype(str) != "completed"]
    if rows.empty:
        return []
    name_column = "module_name" if "module_name" in rows.columns else "plot_name"
    return [
        f"- `{getattr(row, name_column)}`: `{row.status}` ({row.message})"
        for row in rows.itertuples(index=False)
    ]


def _gene_preview(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty or "names" not in frame.columns:
        return "none"
    return _comma_list(frame["names"].astype(str).head(8).tolist())


def _genes_with_threshold(frame: pd.DataFrame, column: str, threshold: int) -> list[str]:
    if column not in frame.columns:
        return []
    return frame.loc[frame[column].fillna(0).astype(int) >= threshold, "gene"].astype(str).tolist()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    display = frame.fillna("n/a").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.values.tolist()]
    return "\n".join([header, separator, *rows])


def _parse_dataset_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 1:
        return f"{numeric:.3f}"
    if numeric == 0:
        return "0.000"
    return f"{numeric:.2e}"


def _comma_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
