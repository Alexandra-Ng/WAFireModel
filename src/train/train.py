"""
Train GCN
Standardize on training years, masked loss, temporal split
Temporal split
- Fit on data.train_years (2014-2018)
- Judge on data.test_years (2019)
- Loss only counts counties whose target is usable ie. mask
- Suppressed counties don't contribute
"""
import numpy as np
import torch

from src.main.config import load_config
from src.main.model import GCN
from src.data.adjacency import load_adjacency, normalize_adjacency
from src.data.dataset import load_dataset, FEATURES


def standardize(data, train_years):
    """Center/scale features and target using TRAIN years only (no test leakage).

    Returns (scaled_data, (y_mean, y_std)) so predictions can be mapped back to
    real rate units later.
    """
    X_train = torch.cat([data[y][0] for y in train_years], dim=0)
    f_mean, f_std = X_train.mean(0), X_train.std(0)
    f_std[f_std == 0] = 1.0                                  # guard constant columns

    y_train = torch.cat([data[y][1][data[y][2]] for y in train_years])  # valid targets
    y_mean, y_std = y_train.mean(), y_train.std()

    scaled = {yr: ((X - f_mean) / f_std, (y - y_mean) / y_std, m)
              for yr, (X, y, m) in data.items()}
    return scaled, (y_mean, y_std)


def masked_mse(pred, target, mask):
    """Mean squared error over usable counties only."""
    return ((pred[mask] - target[mask]) ** 2).mean()


def train_model(cfg, A_hat, data):
    torch.manual_seed(cfg["seed"])
    tcfg = cfg["train"]
    scaled, y_scaler = standardize(data, cfg["data"]["train_years"])

    model = GCN(len(FEATURES), cfg["model"]["hidden_dim"])
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"],
                           weight_decay=tcfg["weight_decay"])

    for _ in range(tcfg["epochs"]):
        model.train()
        opt.zero_grad()
        loss = sum(masked_mse(model(scaled[yr][0], A_hat), scaled[yr][1], scaled[yr][2])
                   for yr in cfg["data"]["train_years"]) / len(cfg["data"]["train_years"])
        loss.backward()
        opt.step()

    return model, scaled, y_scaler


def predict(model, A_hat, scaled, year, y_scaler):
    """Predicted rates for one year, mapped back to real units."""
    y_mean, y_std = y_scaler
    model.eval()
    with torch.no_grad():
        pred = model(scaled[year][0], A_hat) * y_std + y_mean
    return pred


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    counties, A = load_adjacency(cfg["graph"]["adjacency_file"])
    A_hat = normalize_adjacency(A)
    _, data, _ = load_dataset(cfg)

    model, scaled, y_scaler = train_model(cfg, A_hat, data)

    def rmse(year):                          # in real rate units, usable counties only
        _, y, m = data[year]
        pred = predict(model, A_hat, scaled, year, y_scaler)
        return float(torch.sqrt(((pred[m] - y[m]) ** 2).mean()))

    tr = np.mean([rmse(y) for y in cfg["data"]["train_years"]])
    print(f"train RMSE (2014-2018): {tr:.3f}")
    for y in cfg["data"]["test_years"]:
        print(f"test  RMSE ({y}):      {rmse(y):.3f}")

