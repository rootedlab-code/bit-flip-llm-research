"""The published surface: everything in `results/` is published, and described.

These assert a property of the artefacts rather than of the generator, so they hold
whether or not anyone re-runs `kaggle/dataset/build_metadata.py`. The generator already
refused to describe a published file with no descriptor; it did not, until this was
written, notice a results file nobody published at all -- and `e5b-random-manifest.json`
was written, committed and left off the dataset exactly that way. The omission is
invisible from the published page, because every file that is there is documented.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METADATA = ROOT / "kaggle" / "dataset" / "dataset-metadata.json"
DESCRIPTORS = ROOT / "kaggle" / "dataset" / "descriptors.yaml"


@pytest.fixture(scope="module")
def resources() -> list[dict]:
    return json.loads(METADATA.read_text())["resources"]


@pytest.fixture(scope="module")
def descriptors() -> dict:
    return yaml.safe_load(DESCRIPTORS.read_text())


def test_every_results_file_is_published(resources: list[dict]):
    published = {resource["path"] for resource in resources}
    on_disk = {path.name for path in RESULTS.iterdir() if path.is_file()}

    assert not on_disk - published, "in results/ but published nowhere"


def test_every_published_file_exists(resources: list[dict]):
    for resource in resources:
        assert (RESULTS / resource["path"]).exists(), resource["path"]


def test_every_published_file_has_a_descriptor(resources: list[dict], descriptors: dict):
    described = set(descriptors["files"])

    assert not {resource["path"] for resource in resources} - described


def test_every_column_of_every_published_csv_is_described(descriptors: dict):
    """A descriptor table that has silently stopped covering a column is worse than
    none: the gap is invisible on the page while the file still looks documented."""
    for name, entry in descriptors["files"].items():
        columns = entry.get("columns")
        if columns is None:
            continue
        with (RESULTS / name).open() as handle:
            header = next(csv.reader(handle))
        assert not set(header) - set(columns), f"{name}: columns with no descriptor"
