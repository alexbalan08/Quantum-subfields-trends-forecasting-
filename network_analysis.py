import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
Code taken from working notebook and adapted for modular usage using Claude Sonnet 4.6
"""


### GENERAL STATS ####

def network_stats(G: nx.Graph, name: str) -> None:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)
    components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    pct_largest = len(largest_cc) / n * 100
    G_lcc = G.subgraph(largest_cc).copy()
    clustering = nx.average_clustering(G, weight='weight')
    diameter = nx.diameter(G_lcc)
    print(f"{name}: n={n}, e={e}, density={density:.4f}, components={components}, "
          f"pct_largest={pct_largest:.1f}%, clustering={clustering:.4f}, diameter={diameter}")


def betweenness(G: nx.Graph) -> dict[str, float]:
    for u, v, d in G.edges(data=True):
        d['distance'] = 1.0 / d['weight'] if d['weight'] > 0 else 0
    return nx.betweenness_centrality(G, weight='distance', normalized=True)


### COUNTRY GRAPH ###

def top_collaborators_excluding(G: nx.Graph, country: str, exclude: str = None, n: int = 1) -> list:
    neighbors = [
        (nbr, G[country][nbr]['weight'])
        for nbr in G.neighbors(country)
        if nbr != exclude
    ]
    return sorted(neighbors, key=lambda x: x[1], reverse=True)[:n]


def print_top_collaborators(G_country: nx.Graph, countries: list, n: int = 2) -> None:
    col_w = 35
    header = f"{'Country':<{col_w}}"
    for i in range(1, n + 1):
        header += f" {'Top ' + str(i):<{col_w}} {'W':>8}"
    print(header)

    for country in countries:
        row = f"{country:<{col_w}}"
        excluded = None
        for _ in range(n):
            top = top_collaborators_excluding(G_country, country, exclude=excluded, n=1)
            if top:
                row += f" {top[0][0]:<{col_w}} {top[0][1]:>8.2f}"
                excluded = top[0][0]
            else:
                row += f" {'—':<{col_w}} {'—':>8}"
        print(row)


def get_topn_edges_between_countries(G_inst: nx.Graph, country_a: str, country_b: str, n: int = 20) -> None:
    edgelist = [
        (u, v, d["weight"])
        for u, v, d in G_inst.edges(data=True)
        if set([G_inst.nodes[u].get("country"), G_inst.nodes[v].get("country")]) == set([country_a, country_b])
    ]
    edgelist_sorted = sorted(edgelist, key=lambda x: x[2], reverse=True)
    total = sum(w for _, _, w in edgelist_sorted)
    print(f"Total {country_a}-{country_b} weight: {total:.2f}")
    print(f"Number of {country_a}-{country_b} edges: {len(edgelist_sorted)}")
    print(f"\nTop {n}:")
    for u, v, w in edgelist_sorted[:n]:
        print(f"  {u} → {v}: {w:.2f}")


def get_topn_country_collaborations(G_country: nx.Graph, n: int = 20) -> None:
    sorted_edges = sorted(G_country.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    print(f"Top {n} Country Collaboration Pairs:")
    for u, v, data in sorted_edges[:n]:
        print(f"{u:<30} <--> {v:<30} | Weight: {data['weight']:.2f}")


def node_efficiency_impact(G: nx.Graph, node: str, weight: str = 'distance') -> None:
    def efficiency(g):
        total = 0
        n = len(g)
        for source in g.nodes():
            lengths = nx.single_source_dijkstra_path_length(g, source, weight=weight)
            for target, d in lengths.items():
                if source != target and d > 0:
                    total += 1 / d
        return total / (n * (n - 1))

    eff_before = efficiency(G)
    G_removed = G.copy()
    G_removed.remove_node(node)
    eff_after = efficiency(G_removed)
    loss = (eff_before - eff_after) / eff_before
    print(f"Efficiency with {node}: {eff_before:.4f}")
    print(f"Efficiency without {node}: {eff_after:.4f}")
    print(f"Efficiency loss: {loss:.2%}")


def print_efficiency_impact_topn(G_country: nx.Graph, nodes_sorted: list, n: int = 10) -> None:
    for node in nodes_sorted[:n]:
        node_efficiency_impact(G_country, node, weight='distance')
        print()


### INSTITUTION GRAPH ###

def top_international_collabs(G_inst: nx.Graph, country: str, n: int = 5) -> list:
    results = [
        (inst, partner, G_inst.nodes[partner].get("country"), G_inst[inst][partner]['weight'])
        for inst in G_inst.nodes()
        if G_inst.nodes[inst].get("country") == country
        for partner in G_inst.neighbors(inst)
        if G_inst.nodes[partner].get("country") != country
    ]
    return sorted(results, key=lambda x: x[3], reverse=True)[:n]


def get_top_institution_international_collabs(G_inst: nx.Graph, nodes_sorted: list, n: int = 5) -> None:
    for c in nodes_sorted:
        print(f"\n{c}")
        for inst, partner, partner_country, w in top_international_collabs(G_inst, c, n=n):
            print(f"  {inst} → {partner} ({partner_country}): {w:.2f}")


def get_topn_neighbors_inst(G_inst: nx.Graph, node: str, n: int = 20) -> None:
    neighbors = sorted(
        [(nbr, G_inst[node][nbr]["weight"]) for nbr in G_inst.neighbors(node)],
        key=lambda x: x[1],
        reverse=True
    )[:n]
    for nbr, w in neighbors:
        print(f"  {nbr}: {w:.2f}")


def get_top_collaborations_inst(G_inst: nx.Graph, n: int = 20) -> None:
    edges_df = pd.DataFrame([
        {'u': u, 'v': v, 'w': d['weight']}
        for u, v, d in G_inst.edges(data=True)
    ])
    print(edges_df['w'].describe())
    print(f"\nTop {n} Collaborations:")
    print(edges_df.sort_values('w', ascending=False).head(n))


### TOPICAL GRAPH ###

def get_topics_per_inst(target_name: str, df: pd.DataFrame) -> None:
    tgt_rows = df[df['Affiliations_Mapped'].apply(lambda x: target_name in x)]
    topics_count = tgt_rows['Label'].nunique()
    all_topics = tgt_rows['Label'].unique()
    print(f"{target_name} is present in {topics_count} unique topics.")
    print(f"Topics: {sorted(all_topics)}")


def plot_inst_topic_heatmap(G_topic: nx.Graph, inst_nodes: list, topic_weights: list, n: int = 15) -> None:
    inst_totals = {
        inst: sum(G_topic[inst][nbr]['weight'] for nbr in G_topic.neighbors(inst))
        for inst in inst_nodes
    }
    top_insts = sorted(inst_totals, key=inst_totals.get, reverse=True)[:n]
    top_topics = [t for t, _, _ in sorted(topic_weights, key=lambda x: x[1], reverse=True)[:n]]

    matrix = pd.DataFrame(0, index=top_insts, columns=top_topics)
    for inst in top_insts:
        for topic in top_topics:
            if G_topic.has_edge(inst, topic):
                matrix.loc[inst, topic] = G_topic[inst][topic]['weight']

    print(matrix)
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(matrix, cmap='YlOrRd', linewidths=0.5, ax=ax)
    plt.tight_layout()
    plt.show()


def get_topn_inst_per_topic(G_topic: nx.Graph, topic_weights: list, n_topics: int = 15, n_inst: int = 5) -> None:
    for topic in [t for t, _, _ in sorted(topic_weights, key=lambda x: x[1], reverse=True)[:n_topics]]:
        neighbors = sorted(
            G_topic.neighbors(topic),
            key=lambda node: G_topic[topic][node]['weight'],
            reverse=True
        )[:n_inst]
        print(f"{topic}")
        for inst in neighbors:
            print(f"  {inst}: {G_topic[topic][inst]['weight']:.2f}")


def gini(values: np.ndarray) -> float:
    values = sorted(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * sum((i + 1) * v for i, v in enumerate(values))) / (n * cumsum[-1]) - (n + 1) / n


def get_volume_per_topic(G_topic: nx.Graph, topic_weights: list, n: int = None) -> None:
    sorted_topics = sorted(topic_weights, key=lambda x: x[1], reverse=True)
    if n is not None:
        sorted_topics = sorted_topics[:n]
    print(f"{'Topic':<50} | {'Volume':>7} | {'Insts':>6} | {'Mean w':>7} | {'Gini':>6}")
    print("-" * 85)
    for topic, total, n_insts in sorted_topics:
        inst_weights = [G_topic[topic][nbr]['weight'] for nbr in G_topic.neighbors(topic)]
        mean_w = total / n_insts if n_insts > 0 else 0
        g = gini(inst_weights) if len(inst_weights) > 1 else 0
        print(f"{topic:<50} | {total:>7} | {n_insts:>6} | {mean_w:>7.2f} | {g:>6.3f}")
