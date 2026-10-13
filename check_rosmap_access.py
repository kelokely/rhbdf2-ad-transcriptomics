#!/usr/bin/env python3
"""Check local readiness and authorized Synapse access for the ROSMAP cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.ad_cohort_utils import load_cohort_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--cohort", default="ROSMAP_MULTI_REGION")
    parser.add_argument(
        "--output",
        default="results/independent_ad_cohorts/ROSMAP_MULTI_REGION/access_readiness.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    root = Path(registry["_project_root"])
    spec = registry["cohorts"][args.cohort]
    output = Path(args.output)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    result: dict[str, object] = {
        "cohort": args.cohort,
        "access": spec["access"],
        "synapse_accession": spec.get("synapse_accession"),
        "study_accession": spec.get("study_accession"),
        "authorized": False,
        "data_downloaded": False,
        "reason": "not_checked",
    }
    try:
        import synapseclient
    except ImportError:
        result["reason"] = "synapseclient is not installed"
    else:
        token_present = bool(os.getenv("SYNAPSE_AUTH_TOKEN", "").strip())
        config_present = (Path.home() / ".synapseConfig").exists()
        result["credential_source_present"] = token_present or config_present
        if not token_present and not config_present:
            result["reason"] = "No SYNAPSE_AUTH_TOKEN or ~/.synapseConfig is present."
        else:
            try:
                synapse = synapseclient.Synapse(silent=True)
                synapse.login(authToken=os.getenv("SYNAPSE_AUTH_TOKEN") or None, silent=True)
                entity = synapse.get(spec["synapse_accession"], downloadFile=False)
                result["authorized"] = True
                result["entity_name"] = getattr(entity, "name", None)
                result["reason"] = "Authorized metadata access confirmed."
            except Exception as exc:
                result["reason"] = f"Authorization or data-use access check failed: {exc}"
    prepared = Path(spec["prepared_path"])
    prepared = prepared if prepared.is_absolute() else root / prepared
    result["data_downloaded"] = prepared.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
