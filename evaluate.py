import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from train import FEATURES, HitTransformer, load_data, predict


def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0,
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "accuracy": float(accuracy_score(labels, predicted)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def rollout(model, inputs, times, mean, scale, threshold):
    history = []
    probabilities = []
    channel = sum(12 if name == "chroma" else 1 for name in FEATURES[:-1])
    model.eval()
    with torch.inference_mode():
        for features, time_s in zip(inputs, times):
            features = features.copy()
            edges = time_s - 2 + np.linspace(0, 2, 101)
            edges[-1] = time_s
            features[:, channel] = np.diff(np.searchsorted(history, edges, side="left"))
            features = ((np.where(np.isfinite(features), features, mean) - mean) / scale).astype(np.float32)
            probability = model(torch.from_numpy(features[None])).sigmoid().item()
            probabilities.append(probability)
            if probability >= threshold:
                history.append(time_s)
    return np.asarray(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ranked_beat_dataset/manifest.json")
    parser.add_argument("--checkpoint", default="data/ranked_hit_transformer.pt")
    parser.add_argument("--output", default="data/ranked_rollout_results.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES:
        raise ValueError("Checkpoint features do not match the dataset loader")
    frame, x, y = load_data(args.data)
    test_songs = checkpoint.get("test_songs") or [checkpoint["test_song"]]
    indices = frame.loc[frame.song_id.isin(test_songs)].sort_values(["chart_id", "time_s"], kind="stable").index.to_numpy()
    if not len(indices):
        raise ValueError("No test charts match the checkpoint")
    x, y = x[indices], y[indices]
    times = frame.loc[indices, "time_s"].to_numpy()
    mean, scale = checkpoint["mean"].numpy(), checkpoint["scale"].numpy()
    threshold = checkpoint["threshold"]
    model = HitTransformer()
    model.load_state_dict(checkpoint["model"])
    normalized = ((np.where(np.isfinite(x), x, mean) - mean) / scale).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(normalized), torch.from_numpy(y)), batch_size=32)
    real_history = predict(model, loader, torch.device("cpu"))
    chart_ids = frame.loc[indices, "chart_id"].to_numpy()
    predicted_history = np.empty(len(y))
    for chart_id in np.unique(chart_ids):
        mask = chart_ids == chart_id
        predicted_history[mask] = rollout(model, x[mask], times[mask], mean, scale, threshold)
    result = {
        "test_songs": test_songs, "test_charts": np.unique(chart_ids).tolist(), "rows": len(y), "threshold": threshold,
        "real_history": metrics(y, real_history, threshold),
        "predicted_history": metrics(y, predicted_history, threshold),
        "initial_history": "Empty; dataset starts after the first two seconds",
        "feedback": "One Hit per predicted positive timestamp; no lane or chord counts",
        "candidate_times": "Unchanged dataset timestamps, including known Hit times",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    predictions = frame.loc[indices, ["song_id", "chart_id", "time_s"]].copy()
    predictions["actual_hit"] = y.astype(int)
    predictions["probability"] = predicted_history
    predictions["predicted_hit"] = (predicted_history >= threshold).astype(int)
    predictions.to_json(Path(args.output).with_name(Path(args.output).stem + "_predictions.json"), orient="records", indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
