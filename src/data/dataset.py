"""
Parse asthma and wildfire CSVs into yearly graph tensors.

For each year:
    - X: features (39 counties x 5 features)
    - y: target (age-adjusted asthma rate per county)
    - mask: boolean (true when target is usable)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

RATE_COL = "Age Adjusted Rate per 10,000"
FEATURES = ["n_fires", "log_acres", "pct_area", "log_pop", "icd10"]

_NUM = re.compile(f"-?\d+\.?\d*")

def clean_number(cell):
    """
    Clean cell to remove invalid cells
    """
    s = str(cell).strip()
    if s in ("**","--",""):
        return np.nan
    m = _NUM.search(s)
    return float(m.group()) if m else np.nan

def read_year(data_dir, year):
    """
    Load and merge one year csvs and drop state total roll up row
    """
    d = Path(data_dir)
    asthma = pd.read_csv(d / f"Asthma{year}.csv")
    fire = pd.read_csv(d / f"Wildfires{year}.csv")
    df = asthma.merge(fire, on="County")
    df = df[df["County"] != "State Total"]
    return df.sort_values("County").reset_index(drop=True) # Fix county order

def build_year(df, year, icd10_from):
    """
    One years dataframe (X,y,mask) tensors
    """
    rate = df[RATE_COL].map(clean_number).to_numpy(np.float32)
    pop = df["Population"].map(clean_number).to_numpy(np.float32)
    n_fires = df["# of Fires"].map(clean_number).fillna(0).to_numpy(np.float32)
    acres = df["Acres Burned"].map(clean_number).fillna(0).to_numpy(np.float32)
    pct = df["Percent Area Burned"].map(clean_number).fillna(0).to_numpy(np.float32)
    feats = np.stack([
        n_fires,
        np.log1p(acres),                                 # acres heavy-tailed -> log
        pct,
        np.log(pop),                                     # population -> log
        np.full_like(pop, float(year >= icd10_from)),    # ICD era: 0 (2014), 1 (2016+)
    ], axis=1)
    X = torch.tensor(feats)
    y = torch.tensor(rate)
    mask = ~torch.isnan(y)                               # usable targets only
    return X, y, mask

def load_dataset(cfg):
    """
    Build dataset from config dict.
    node set comes from the graph's adjacency file, data rows line
    up with A_hat rows by construction, reindex() selects those 39
    counties in that order and drops everything else
 
    Returns (counties, data, FEATURES) with data[year] = (X, y, mask).
    """
    from src.data.adjacency import load_adjacency
    dc = cfg["data"]
    counties, _ = load_adjacency(cfg["graph"]["adjacency_file"])   # canonical 39, sorted
    data = {}
    for year in dc["years"]:
        df = read_year(dc["dir"], year)
        df = df.set_index("County").reindex(counties).reset_index()
        data[year] = build_year(df, year, dc["icd10_from"])
    return counties, data, FEATURES

if __name__ == "__main__":
    from src.main.config import load_config
    cfg = load_config("configs/default.yaml")
    counties, data, feats = load_dataset(cfg)
    print(f"{len(counties)} counties, {len(feats)} features: {feats}")
    for year, (X, y, mask) in data.items():
        print(f"  {year}: X{tuple(X.shape)}  valid targets {int(mask.sum())}/{len(y)}")

















