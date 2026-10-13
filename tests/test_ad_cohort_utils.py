"""Tests for independent AD cohort registry and metadata helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest

from src.ad_cohort_utils import (
    _safe_unique_column_names,
    build_registry_status,
    parse_geo_miniml_metadata,
)


class ADCohortUtilsTests(unittest.TestCase):
    def test_parse_geo_miniml_metadata(self) -> None:
        xml = b"""<?xml version="1.0"?>
<MINiML xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML">
  <Sample iid="GSM1">
    <Title>GEX_D1</Title>
    <Channel position="1">
      <Characteristics tag="pathological diagnosis">AD</Characteristics>
      <Characteristics tag="genotype">E3/E4</Characteristics>
    </Channel>
    <Supplementary-Data type="TAR">ftp://example/GSM1_GEX_D1.tar.gz</Supplementary-Data>
  </Sample>
</MINiML>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "family.xml.tgz"
            with tarfile.open(path, mode="w:gz") as archive:
                info = tarfile.TarInfo("family.xml")
                info.size = len(xml)
                archive.addfile(info, BytesIO(xml))
            result = parse_geo_miniml_metadata(path)
        self.assertEqual(result.loc[0, "geo_accession"], "GSM1")
        self.assertEqual(result.loc[0, "geo_pathological_diagnosis"], "AD")
        self.assertEqual(result.loc[0, "geo_genotype"], "E3/E4")

    def test_registry_status_rejects_aria2_sparse_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "large.h5ad"
            source.write_bytes(b"12345")
            source.with_name(source.name + ".aria2").write_text("incomplete", encoding="utf-8")
            registry = {
                "_project_root": str(root),
                "cohorts": {
                    "TEST": {
                        "title": "Test",
                        "access": "public",
                        "role": "test",
                        "source_directory": "source",
                        "prepared_path": "source/large.h5ad",
                        "sources": [
                            {
                                "filename": "large.h5ad",
                                "expected_size_bytes": 5,
                            }
                        ],
                    }
                },
            }
            status = build_registry_status(registry)
        self.assertFalse(bool(status.loc[0, "all_sources_present"]))
        self.assertFalse(bool(status.loc[0, "prepared_object_present"]))

    def test_safe_unique_column_names_are_hdf5_compatible(self) -> None:
        result = _safe_unique_column_names(["average UMI/cell", "A value", "A/value", "///"])
        self.assertEqual(result, ["average_umi_cell", "a_value", "a_value_2", "unnamed"])


if __name__ == "__main__":
    unittest.main()
