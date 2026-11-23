from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from .data import NetworkInstance


def plot_network(instance: NetworkInstance, solution: pd.DataFrame, path: Path, title: str, failed_route: tuple[str, str] | None = None) -> None:
    G = nx.DiGraph()
    wh_ids = instance.warehouses["warehouse_id"].tolist()
    st_ids = instance.stores["store_id"].tolist()
    for w in wh_ids:
        G.add_node(w, kind="warehouse")
    for s in st_ids:
        G.add_node(s, kind="store")

    pos = {}
    for k, w in enumerate(wh_ids):
        pos[w] = (0.0, 1.0 - k / max(1, len(wh_ids) - 1))
    for k, s in enumerate(st_ids):
        pos[s] = (1.0, 1.0 - k / max(1, len(st_ids) - 1))

    sol_map = solution.set_index(["warehouse_id", "store_id"]) if not solution.empty else pd.DataFrame()
    used_edges = []
    unused_edges = []
    forbidden_edges = []
    labels = {}

    for row in instance.routes.itertuples(index=False):
        edge = (row.warehouse_id, row.store_id)
        if not row.allowed or (failed_route is not None and edge == failed_route):
            forbidden_edges.append(edge)
            G.add_edge(*edge)
            continue
        G.add_edge(*edge)
        flow = 0.0
        if not solution.empty and edge in sol_map.index:
            flow = float(sol_map.loc[edge, "flow_units"])
        if flow > 1e-7:
            used_edges.append(edge)
        else:
            unused_edges.append(edge)
        if flow > 1e-7:
            labels[edge] = f"{flow:.1f}/{row.capacity_units:.1f}"

    fig, ax = plt.subplots(figsize=(13, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=wh_ids, node_size=1500, node_shape="s", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=st_ids, node_size=1500, node_shape="o", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=unused_edges, width=0.7, alpha=0.25, arrows=True, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=used_edges, width=2.0, arrows=True, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=forbidden_edges, width=1.0, style="dashed", alpha=0.25, arrows=False, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7, rotate=False, label_pos=0.5, ax=ax)
    ax.set_title(title + "\nподпись используемой дуги: поток/пропускная способность")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_costs(scenarios: pd.DataFrame, path: Path) -> None:
    shown = scenarios[scenarios["objective_cost"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(shown["scenario"], shown["objective_cost"])
    ax.set_ylabel("Суммарная стоимость")
    ax.set_xlabel("Сценарий")
    ax.set_title("Стоимость допустимых сценариев")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bottlenecks(bottlenecks: pd.DataFrame, path: Path) -> None:
    labels = bottlenecks["warehouse_id"] + "→" + bottlenecks["store_id"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, bottlenecks["capacity_shadow_benefit"])
    ax.set_ylabel("Снижение стоимости при +1 ед. пропускной способности")
    ax.set_title("Теневые цены насыщенных маршрутов")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_optimal_vs_greedy(optimal_cost: float, greedy_cost: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(["HiGHS", "Жадная эвристика"], [optimal_cost, greedy_cost])
    ax.set_ylabel("Суммарная стоимость")
    ax.set_title("Оптимальное решение и жадный алгоритм")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lp_vs_milp(lp_cost: float, milp_cost: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(["LP", "MILP"], [lp_cost, milp_cost])
    ax.set_ylabel("Суммарная стоимость")
    ax.set_title("Доп. задание: непрерывный и целочисленный планы")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
