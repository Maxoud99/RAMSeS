"""
TSB-AD file names and entity ids, without pandas or numpy.

TSB-AD (Liu & Paparrizos, "The Elephant in the Room: Towards A Reliable Time-
Series Anomaly Detection Benchmark", NeurIPS 2024 Datasets & Benchmarks;
https://thedatumorg.github.io/TSB-AD/) ships two collections of CSV files,
TSB-AD-U (870 univariate series) and TSB-AD-M (200 multivariate series). Each
file is one series, named

    [index]_[Dataset]_id_[id]_[Domain]_tr_[TrainEnd]_1st_[FirstAnomaly].csv
    e.g. 001_NAB_id_1_Facility_tr_1007_1st_2014.csv

with one column per channel plus `Label` (0/1, one per sample). Samples
[0, TrainEnd) are the training part, the rest is the test part with its real
labels. The benchmark calls the training part anomaly-free; it is not always:
177 TSB-AD-U and 22 TSB-AD-M series label samples before TrainEnd (mostly
YAHOO, Stock and TAO). The split is taken from the file name all the same, as
TSB-AD itself does, and the detectors train on that part unsupervised.

Kept apart from `Datasets/load.py` so the web UI's catalog can list 870
entities by name without importing pandas and sklearn.

Entity ids
----------
An entity is the file name up to `_tr_`: `001_NAB_id_1_Facility`. That is
unique (the leading index alone already is), stable across releases that keep
the index, and reads as what the series is. The trailing `_tr_.._1st_..`
fields are facts ABOUT the series, not its name, which is how the loader for
UCR's anomaly archive treats its trailing index fields too. The full stem and
the file name are accepted as aliases and resolve to the same id
(`canonical_entity`), so a results or checkpoint directory is never split
between two spellings of one series.
"""

import os
import re
from typing import Dict, Optional, Tuple

# Dataset key (what --dataset takes, lower case) -> directory under the data
# root. Directory lookup is case-insensitive, as for every other dataset.
TSB_AD_DIRECTORIES: Dict[str, str] = {"tsb-ad-u": "TSB-AD-U", "tsb-ad-m": "TSB-AD-M"}

_NAME = re.compile(r"^(?P<index>\d+)_(?P<source>.+?)_id_(?P<id>\d+)_(?P<domain>.+)"
                   r"_tr_(?P<train_end>\d+)_1st_(?P<first_anomaly>\d+)$")


def is_tsb_ad(dataset) -> bool:
    return str(dataset).lower() in TSB_AD_DIRECTORIES


def parse_name(name: str) -> Optional[Dict[str, object]]:
    """The fields of a TSB-AD file name (with or without `.csv`), or None.

    >>> parse_name("001_NAB_id_1_Facility_tr_1007_1st_2014.csv")["train_end"]
    1007
    """
    stem = str(name)
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    match = _NAME.match(stem)
    if match is None:
        return None
    fields = match.groupdict()
    return {
        "stem": stem,
        "entity": stem[:stem.rindex("_tr_")],
        "index": fields["index"],
        "source": fields["source"],
        "id": int(fields["id"]),
        "domain": fields["domain"],
        "train_end": int(fields["train_end"]),
        "first_anomaly": int(fields["first_anomaly"]),
    }


def entity_from_filename(name: str) -> Optional[str]:
    """'001_NAB_id_1_Facility_tr_1007_1st_2014.csv' -> '001_NAB_id_1_Facility'."""
    if not str(name).lower().endswith(".csv"):
        return None
    meta = parse_name(name)
    return meta["entity"] if meta else None


def resolve_directory(root_dir, dataset) -> Optional[str]:
    """The dataset's directory under `root_dir`, matched case-insensitively."""
    wanted = TSB_AD_DIRECTORIES.get(str(dataset).lower(), str(dataset))
    exact = os.path.join(str(root_dir), wanted)
    if os.path.isdir(exact):
        return exact
    try:
        names = os.listdir(str(root_dir))
    except OSError:
        return None
    for name in names:
        if name.lower() == wanted.lower() and os.path.isdir(os.path.join(str(root_dir), name)):
            return os.path.join(str(root_dir), name)
    return None


# directory -> (mtime, {entity id: file name}). 870 names are parsed once per
# change of the directory, not on every lookup.
_INDEX: Dict[str, Tuple[float, Dict[str, str]]] = {}


def index(directory: str) -> Dict[str, str]:
    """{entity id: file name} of one TSB-AD directory."""
    try:
        mtime = os.stat(directory).st_mtime
    except OSError:
        return {}
    cached = _INDEX.get(directory)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    found: Dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        entity = entity_from_filename(name)
        if entity is not None:
            found[entity] = name
    _INDEX[directory] = (mtime, found)
    return found


def _lookup_key(entity) -> str:
    """An entity id, full stem or file name -> the id it names."""
    text = str(entity[0] if isinstance(entity, (list, tuple)) else entity).strip()
    meta = parse_name(text)
    return meta["entity"] if meta else text


def find_file(root_dir, dataset, entity) -> Tuple[str, Dict[str, object]]:
    """(path, parsed name) of `entity`'s CSV. Raises FileNotFoundError."""
    directory = resolve_directory(root_dir, dataset)
    if directory is None:
        raise FileNotFoundError(
            f"No {TSB_AD_DIRECTORIES.get(str(dataset).lower(), dataset)} directory under "
            f"{root_dir}")
    key = _lookup_key(entity)
    files = index(directory)
    name = files.get(key)
    if name is None:
        # A bare index ("1", "001") names one series too; accepted because it
        # is what the TSB-AD tables print.
        if key.isdigit():
            matches = [f for e, f in files.items() if int(e.split("_", 1)[0]) == int(key)]
            if len(matches) == 1:
                name = matches[0]
    if name is None:
        raise FileNotFoundError(f"No TSB-AD series '{entity}' in {directory}")
    return os.path.join(directory, name), parse_name(name)


def canonical_entity(root_dir, dataset, entity) -> str:
    """The entity id that `entity` (id, stem, file name or index) resolves to;
    `entity` itself when it resolves to nothing, so the loader reports why."""
    try:
        return find_file(root_dir, dataset, entity)[1]["entity"]
    except (FileNotFoundError, OSError):
        return str(entity)
