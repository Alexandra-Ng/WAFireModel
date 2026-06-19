"""
Visualization of the GCN experiment. Writes PNGs to figures/.
fig1_county_graph : the adjacency graph, nodes coloured by 2019 rate
fig2_diagnostics  : target missingness, model comparison, pred vs actual
"""
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.main.config import load_config
from src.data.adjacency import load_adjacency, normalize_adjacency
from src.data.dataset import load_dataset
from src.evaluate.evaluate import evaluate

OUT = Path("figures")


def main():
    cfg = load_config("configs/default.yaml")
    counties, A = load_adjacency(cfg["graph"]["adjacency_file"])
    A_hat = normalize_adjacency(A)
    _, data, _ = load_dataset(cfg)
    rows, preds = evaluate(cfg, A_hat, data)
    years = cfg["data"]["years"]
    test_year = cfg["data"]["test_years"][0]

    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.titleweight": "bold"})

    # ---- fig 1: the graph, coloured by test-year actual rate ----
    G = nx.Graph()
    G.add_nodes_from(counties)
    for i, ci in enumerate(counties):
        for j, cj in enumerate(counties):
            if i < j and A[i, j]:
                G.add_edge(ci, cj)
    pos = nx.kamada_kawai_layout(G)

    y_actual = data[test_year][1].numpy()
    valid = data[test_year][2].numpy()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc")
    nx.draw_networkx_nodes(G, pos, nodelist=[c for c, v in zip(counties, valid) if not v],
                           node_color="#e0e0e0", edgecolors="#999", node_size=420, ax=ax)
    sc = nx.draw_networkx_nodes(
        G, pos, nodelist=[c for c, v in zip(counties, valid) if v],
        node_color=y_actual[valid], cmap="YlOrRd", node_size=520, edgecolors="#444", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6.5, ax=ax)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=f"{test_year} asthma rate / 10k")
    ax.set_title(f"WA county graph, coloured by {test_year} rate\n"
                 "grey = suppressed  ·  topology layout, not geographic")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_county_graph.png", bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: missingness | model comparison | pred vs actual ----
    fig = plt.figure(figsize=(13, 4.4))
    gs = GridSpec(1, 3, width_ratios=[1.05, 0.8, 1.0], wspace=0.32)

    ax0 = fig.add_subplot(gs[0])
    M = np.array([[not data[yr][2][i] for yr in years] for i in range(len(counties))])
    ax0.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax0.set_xticks(range(len(years))); ax0.set_xticklabels(years)
    ax0.set_yticks(range(len(counties))); ax0.set_yticklabels(counties, fontsize=5.5)
    ax0.set_title("Target missingness\n(black = suppressed)")

    ax1 = fig.add_subplot(gs[1])
    order = sorted(rows, key=lambda r: r[1])
    names = [r[0] for r in order]; rmses = [r[1] for r in order]
    colors = ["#d62728" if n == "GCN" else "#bbb" for n in names]
    ax1.barh(range(len(names)), rmses, color=colors)
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=8)
    ax1.axvline(y_actual[valid].std(), ls="--", color="k", lw=1)
    ax1.set_xlabel("Test RMSE"); ax1.set_title("Model comparison\n(dashed = target std)")
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(gs[2])
    gcn = preds["GCN"][test_year][valid]
    rdg = preds["ridge"][test_year][valid]
    act = y_actual[valid]
    ax2.scatter(act, rdg, c="#7f7f7f", s=34, label="ridge")
    ax2.scatter(act, gcn, c="#d62728", s=34, label="GCN")
    lim = [0, max(act.max(), gcn.max()) * 1.05]
    ax2.plot(lim, lim, "k--", lw=1); ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("actual rate / 10k"); ax2.set_ylabel("predicted")
    ax2.set_title(f"{test_year} predicted vs actual"); ax2.legend(fontsize=7)

    fig.savefig(OUT / "fig2_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig1_county_graph.png and {OUT}/fig2_diagnostics.png")


if __name__ == "__main__":
    main()
