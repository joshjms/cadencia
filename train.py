import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cadencia.dataset import FEATURES, load_dataset


def load_data(path):
    path = Path(path)
    if path.is_dir() or path.name == "manifest.json":
        frame, x, y = load_dataset(path)
        if x.shape[1:] != (100, 20):
            raise ValueError(f"Expected 100 steps and 20 features, got {x.shape[1:]}")
        return frame, x, y
    frame = pd.read_json(path)
    arrays = []
    for name in FEATURES:
        values = np.asarray(frame[name].tolist(), dtype=np.float32)
        arrays.append(values[..., None] if values.ndim == 2 else values)
    x = np.concatenate(arrays, axis=-1)
    y = np.asarray(frame.hit.tolist(), dtype=np.float32)[:, 1]
    if x.shape[1:] != (100, 20):
        raise ValueError(f"Expected 100 steps and 20 features, got {x.shape[1:]}")
    return frame, x, y


class HitTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(20, 32)
        self.position = nn.Parameter(torch.randn(1, 100, 32) * 0.02)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=128, dropout=0.1,
                batch_first=True,
            ) for _ in range(2)
        ])
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.projection(x) + self.position
        for layer in self.encoder:
            x = layer(x)
        return self.output(x.mean(dim=1)).squeeze(-1)


def predict(model, loader, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([
            model(x.to(device)).sigmoid().cpu() for x, _ in loader
        ]).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ranked_beat_dataset/manifest.json")
    parser.add_argument("--output", default="data/ranked_hit_transformer.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    frame, x, y = load_data(args.data)
    if "split" in frame:
        splits = [(frame["split"] == name).to_numpy() for name in ("train", "validation", "test")]
        if not all(mask.any() for mask in splits):
            raise ValueError("At least three independent groups are needed for separate splits")
    else:
        songs = rng.permutation(frame.song_id.unique())
        if len(songs) < 3:
            raise ValueError("At least three songs are needed for separate splits")
        splits = [frame.song_id.isin(songs[:-2]).to_numpy(),
                  (frame.song_id == songs[-2]).to_numpy(),
                  (frame.song_id == songs[-1]).to_numpy()]
    split_songs = [sorted(int(song) for song in frame.song_id[mask].unique()) for mask in splits]
    train = x[splits[0]]
    train[~np.isfinite(train)] = np.nan
    mean = np.nanmean(train, axis=(0, 1))
    mean = np.nan_to_num(mean, nan=0.0)
    scale = np.nanstd(train, axis=(0, 1))
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    del train
    x = np.where(np.isfinite(x), x, mean)
    x -= mean
    x /= scale
    loaders = [DataLoader(
        TensorDataset(torch.from_numpy(x[mask]), torch.from_numpy(y[mask])),
        batch_size=args.batch_size, shuffle=i == 0,
    ) for i, mask in enumerate(splits)]
    del x
    for name, mask in zip(["train", "validation", "test"], splits):
        print(f"{name}: songs={frame.song_id[mask].unique().tolist()}, rows={mask.sum()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HitTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    best_score, stale = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for features, labels in loaders[0]:
            optimizer.zero_grad()
            loss = loss_fn(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        probabilities = predict(model, loaders[1], device)
        score = average_precision_score(y[splits[1]], probabilities)
        print(f"epoch={epoch + 1} loss={total / splits[0].sum():.4f} validation_AP={score:.4f}")
        if score > best_score:
            best_score, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 7:
                break
    model.load_state_dict(best_state)
    validation = predict(model, loaders[1], device)
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = [precision_recall_fscore_support(
        y[splits[1]], validation >= t, average="binary", zero_division=0,
    )[2] for t in thresholds]
    threshold = float(thresholds[np.argmax(scores)])
    probabilities = predict(model, loaders[2], device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y[splits[2]], probabilities >= threshold, average="binary", zero_division=0,
    )
    ap = average_precision_score(y[splits[2]], probabilities)
    print(f"test: AP={ap:.4f} precision={precision:.4f} recall={recall:.4f} F1={f1:.4f} threshold={threshold:.2f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {k: v.cpu() for k, v in best_state.items()},
        "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
        "threshold": threshold, "features": FEATURES,
        "train_songs": split_songs[0], "validation_songs": split_songs[1], "test_songs": split_songs[2],
        "validation_song": split_songs[1][0], "test_song": split_songs[2][0],
        "seed": args.seed,
    }, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
