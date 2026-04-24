import csv
import os
from typing import Dict, Any

CSV_HEADER = [
    "file_id",
    "label",
    "score_simple_energy",
    "score_simple_spectral",
    "score_aasist",
    "score_rawnet2",
    "score_ens",
    "verdict",
    "duration_sec",
    "n_windows",
    "runtime_sec",
]


def append_experiment_row(csv_path: str, row: Dict[str, Any]) -> None:
    dirname = os.path.dirname(csv_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})
