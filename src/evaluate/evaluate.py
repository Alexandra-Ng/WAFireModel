"""
Compare GCN against simple baselines on the temporal holdout

baselines use the same standardized features and the same train/test split:
  mean          : predict the train-mean rate (era-matched) for every county
  ridge         : linear regression on node features, NO graph
  ridge+spatial : ridge on features + neighbour-averaged features (A_hat @ X)
  GCN           : the model

"""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.main.config import load_config
from src.data.adjacency import load_adjacency, normalize_adjacency
from src.data.dataset import load_dataset
from src.train.train import train_model, predict


def _stack(scaled, years, A_hat=None):
    """
    Stack usable (X, y) rows over years
    optionally append neighbour avg feats
    """
    Xs, ys = [], []
    for yr in years:
        X, y, m = scaled[yr]
        feat = X if A_hat is None else torch.cat([X, A_hat @ X], dim=1)
        Xs.append(feat[m]); ys.append(y[m])
    return torch.cat(Xs).numpy(), torch.cat(ys).numpy()


def evaluate(cfg, A_hat, data):
    model, scaled, y_scaler = train_model(cfg, A_hat, data)
    y_mean, y_std = (s.item() for s in y_scaler)
    train_years = cfg["data"]["train_years"]
    test_years = cfg["data"]["test_years"]
    icd = cfg["data"]["icd10_from"]

    preds = {}   # name -> {year -> full (N,) prediction array in real units}

    # mean baseline, matched to the test era (2014 ICD-9 rates are ~2x higher)
    test_era = test_years[0] >= icd
    era_years = [y for y in train_years if (y >= icd) == test_era]
    mean_rate = torch.cat([data[y][1][data[y][2]] for y in era_years]).mean().item()
    preds["mean"] = {yr: np.full(data[yr][1].shape[0], mean_rate) for yr in test_years}

    # ridge on features only
    Xtr, ytr = _stack(scaled, train_years)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    preds["ridge"] = {yr: ridge.predict(scaled[yr][0].numpy()) * y_std + y_mean
                      for yr in test_years}

    # ridge on features + spatial lag (graph as a feature smoother)
    Xtr2, ytr2 = _stack(scaled, train_years, A_hat)
    ridge2 = Ridge(alpha=1.0).fit(Xtr2, ytr2)
    preds["ridge+spatial"] = {
        yr: ridge2.predict(torch.cat([scaled[yr][0], A_hat @ scaled[yr][0]], 1).numpy())
            * y_std + y_mean
        for yr in test_years}

    # the GCN
    preds["GCN"] = {yr: predict(model, A_hat, scaled, yr, y_scaler).numpy()
                    for yr in test_years}

    # score each on usable test counties
    rows = []
    for name, by_year in preds.items():
        yt = np.concatenate([data[yr][1][data[yr][2]].numpy() for yr in test_years])
        pt = np.concatenate([by_year[yr][data[yr][2].numpy()] for yr in test_years])
        rows.append((name, np.sqrt(mean_squared_error(yt, pt)), mean_absolute_error(yt, pt)))
    return rows, preds


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    _, A = load_adjacency(cfg["graph"]["adjacency_file"])
    A_hat = normalize_adjacency(A)
    _, data, _ = load_dataset(cfg)
    rows, _ = evaluate(cfg, A_hat, data)

    print(f"{'model':<16}{'RMSE':>8}{'MAE':>8}   (test 2019, lower is better)")
    for name, rmse, mae in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<16}{rmse:>8.3f}{mae:>8.3f}")
