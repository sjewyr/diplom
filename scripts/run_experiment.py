from __future__ import annotations
import os
import glob
import time
from dataclasses import dataclass
from typing import List, Tuple

from app.config import AppConfig
from app.pipeline import analyze_file
from app.logging_utils import append_experiment_row


@dataclass
class Item:
    path: str
    label: int  # 0=real, 1=spoof


def load_items(real_dir: str, spoof_dir: str) -> List[Item]:
    items: List[Item] = []

    for p in glob.glob(os.path.join(real_dir, "*")):
        if os.path.isfile(p):
            items.append(Item(path=p, label=0))

    for p in glob.glob(os.path.join(spoof_dir, "*")):
        if os.path.isfile(p):
            items.append(Item(path=p, label=1))

    return items


def predict_label(verdict: str) -> int:
    return 1 if verdict == "spoof_detected" else 0


def confusion(items: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for y_true, y_pred in items:
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    return tp, tn, fp, fn


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = tp + tn + fp + fn
    return 0.0 if denom == 0 else 100.0 * (tp + tn) / denom


def eer_from_scores(scores_labels: List[Tuple[float, int]], n_thr: int = 500) -> float:
    if not scores_labels:
        return 0.0

    scores = [s for s, _ in scores_labels]
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return 0.0

    best_diff = 1e9
    best_eer = 0.0

    for k in range(n_thr + 1):
        thr = lo + (hi - lo) * k / n_thr

        fp = fn = tp = tn = 0
        for s, y in scores_labels:
            y_pred = 1 if s >= thr else 0

            if y == 1 and y_pred == 1:
                tp += 1
            elif y == 0 and y_pred == 0:
                tn += 1
            elif y == 0 and y_pred == 1:
                fp += 1
            elif y == 1 and y_pred == 0:
                fn += 1

        far = fp / (fp + tn + 1e-12)
        frr = fn / (fn + tp + 1e-12)
        diff = abs(far - frr)
        eer = (far + frr) / 2.0

        if diff < best_diff:
            best_diff = diff
            best_eer = eer

    return float(best_eer * 100.0)


def main():
    real_dir = "data/processed/real"
    spoof_dir = "data/processed/spoof"
    csv_path = "runs/experiment.csv"

    cfg = AppConfig()

    items = load_items(real_dir, spoof_dir)
    if not items:
        print("Нет файлов для эксперимента.")
        return

    y_pairs: List[Tuple[int, int]] = []
    scores_labels: List[Tuple[float, int]] = []

    for it in items:
        t0 = time.time()
        res = analyze_file(it.path, os.path.basename(it.path), cfg)
        runtime = time.time() - t0

        y_pred = predict_label(res["verdict"])
        y_pairs.append((it.label, y_pred))

        score_ens = float(res["scores"]["record_score"])
        scores_labels.append((score_ens, it.label))

        ms = res.get("model_scores", {})

        row = {
            "file_id": os.path.basename(it.path),
            "label": it.label,
            "score_simple_energy": ms.get("simple_energy", ""),
            "score_simple_spectral": ms.get("simple_spectral", ""),
            "score_aasist": ms.get("aasist", ""),
            "score_rawnet2": ms.get("rawnet2", ""),
            "score_ens": score_ens,
            "verdict": res["verdict"],
            "duration_sec": res.get("duration_sec", ""),
            "n_windows": res.get("n_windows", ""),
            "runtime_sec": round(runtime, 4),
        }
        append_experiment_row(csv_path, row)

    tp, tn, fp, fn = confusion(y_pairs)
    acc = accuracy(tp, tn, fp, fn)
    eer = eer_from_scores(scores_labels)

    print("TP, TN, FP, FN:", tp, tn, fp, fn)
    print("Accuracy (%):", round(acc, 2))
    print("Approx EER (%):", round(eer, 2))
    print("CSV saved to:", csv_path)


if __name__ == "__main__":
    main()
