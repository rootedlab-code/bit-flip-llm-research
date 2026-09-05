"""Build `dataset-metadata.json` from `descriptors.yaml` and the published CSVs.

Kaggle's usability score is driven by four things this file can supply — a subtitle, a
tag list, a description per file and a description per column — and by one it cannot: a
cover image, which is settable only from the web interface.

Three of Kaggle's rules are enforced here because each one costs a failed update to
discover, and none of them is reported at the point of use:

- a file left out of `resources` is simply not published, and nothing says so: the
  omission is invisible on the page and in the exit status. `e5b-random-manifest.json`
  was written, committed and left unpublished exactly that way;
- the subtitle must be 20 to 80 characters, or the whole update is refused;
- keywords come from a controlled vocabulary, and one unknown keyword fails the update;
- **`kaggle datasets version` does not send the licence at all.** It uploads files. Only
  `kaggle datasets metadata --update` carries title, subtitle, description, licence and
  keywords, and the difference is invisible: pushing a version with a licence in the file
  exits 0 and leaves the page saying `unknown`.

Usage:  uv run python kaggle/dataset/build_metadata.py
"""

from __future__ import annotations

import csv
import json
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
RESULTS = HERE.parent.parent / "results"
DESCRIPTORS = HERE / "descriptors.yaml"
TARGET = HERE / "dataset-metadata.json"
DESCRIPTION = HERE / "DESCRIPTION.md"

SUBTITLE_LIMITS = (20, 80)


class MetadataError(ValueError):
    """The metadata cannot be built from what is on disk."""


def unpublished(resources: list[dict]) -> list[str]:
    """Files under `results/` that no resource entry publishes.

    `build` checks that everything published is described. This checks the other
    direction, which is the one that fails silently: a results file nobody listed is
    absent from the dataset with no error anywhere, and the page looks complete because
    every file that *is* there is documented.
    """
    published = {resource["path"] for resource in resources}
    on_disk = {path.name for path in RESULTS.iterdir() if path.is_file()}
    return sorted(on_disk - published)


def build(descriptors: dict, resources: list[dict]) -> list[dict]:
    """One resource entry per published file, with its column schema.

    Raises rather than emitting a partial schema: a descriptor table that has silently
    stopped covering a column is worse than none, because the gap is invisible on the
    page while the file still looks documented.
    """
    files = descriptors["files"]
    built = []
    for resource in resources:
        path = resource["path"]
        if path not in files:
            raise MetadataError(
                f"{path} is published with no entry in {DESCRIPTORS.name}"
            )
        entry: dict = {"path": path, "description": files[path]["description"]}
        columns = files[path].get("columns")
        if columns:
            with (RESULTS / path).open() as handle:
                header = next(csv.reader(handle))
            missing = [name for name in header if name not in columns]
            if missing:
                raise MetadataError(f"{path}: columns with no descriptor: {missing}")
            entry["schema"] = {
                "fields": [{"name": n, "description": columns[n]} for n in header]
            }
        built.append(entry)
    return built


def main() -> int:
    descriptors = yaml.safe_load(DESCRIPTORS.read_text())
    subtitle = descriptors["subtitle"]
    low, high = SUBTITLE_LIMITS
    if not low <= len(subtitle) <= high:
        raise MetadataError(
            f"subtitle is {len(subtitle)} characters, Kaggle wants {low}-{high}"
        )

    meta = json.loads(TARGET.read_text())
    meta["subtitle"] = subtitle
    meta["description"] = DESCRIPTION.read_text()
    meta["keywords"] = descriptors["keywords"]
    # Under the canonical name the server returns, not the shorthand it accepts: two
    # spellings of one licence make a diff between the repository and the page look like
    # a defect when it is not.
    meta["licenses"] = [{"name": descriptors["licence"]}]
    missing = unpublished(meta["resources"])
    if missing:
        raise MetadataError(f"in {RESULTS.name}/ but published nowhere: {missing}")
    meta["resources"] = build(descriptors, meta["resources"])

    TARGET.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    columns = sum(len(r.get("schema", {}).get("fields", [])) for r in meta["resources"])
    print(f"{len(meta['resources'])} files, {columns} columns described -> {TARGET.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
