import pandas as pd
import networkx as nx
from typing import Dict
from itertools import combinations


# constants
PATH_TO_CANON_MAP = ""
PATH_TO_DF = ""  # in .csv!!!!!!!


def build_inverse_mapping(input_dict: Dict[str, str]) -> Dict[str, str]:  # inverts the first mapping
    inv_map = {}
    for k, v in input_dict.items():
        for aff in v:
            inv_map[aff] = k
    return inv_map

def newman_weight(n: int) -> int:
    return 1 / (n - 1)

def map_raw_aff(df: pd.DataFrame, map_names: Dict[str, str]) -> pd.DataFrame:
    inv_map: dict = build_inverse_mapping(map_names)
    df["Affiliations"] = (df["Affiliations"].fillna("").apply(lambda x: [a.strip() for a in x.split(";") if a.strip()]))
    df["Affiliations_Mapped"] = df["Affiliations"].apply(lambda lst: [inv_map.get(a, a) for a in lst])
    return df

def build_G_inst(df: pd.DataFrame) -> nx.Graph:  # pass in the CLEAN dataset, with map applied already
    G = nx.Graph()
    try:
        for aff_list in df["Affiliations_Mapped"]:
            unique_affs = list(set(aff_list))
            N = len(unique_affs)
            G.add_nodes_from(unique_affs)
            if N >= 2:  # only compute weight for collaboration, singletons don't need it
                w = newman_weight(N)
                for u, v in combinations(unique_affs, 2):
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += w
                    else:
                        G.add_edge(u, v, weight=w)
        print("Graph succesfully created!")
        print("Nodes:", G.number_of_nodes())
        print("Edges:", G.number_of_edges())
    except Exception as e:
        print(f"Error while building graph: {e}")

def build_G_topic(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for idx, row in df[['Label', 'Affiliations_Mapped']].iterrows():
        topic = row['Label']
        affs = row['Affiliations_Mapped']
        if not topic or not isinstance(affs, list):
            continue
        for inst in set(affs):
            if G.has_edge(inst, topic):
                G[inst][topic]['weight'] += 1
            else:
                G.add_edge(inst, topic, weight=1)
                G.nodes[inst]['type'] = 'institution'
                G.nodes[topic]['type'] = 'topic'
    return G

def build_G_country(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for countries in df["Country"]:
        if not isinstance(countries, list):
            continue
        unique_countries = list(set(countries))
        G.add_nodes_from(unique_countries)
        if len(unique_countries) >= 2:
            for u, v in combinations(unique_countries, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
    return G

def get_neighbors(name_of_org: str, G: nx.Graph, n_neighbors_to_display: int) -> None:  # it just prints
    # EXAMPLE USAGE: name_of_org = "ETH Zurich" // will try later on to have a system to recognize incorrect spellings
    print(sorted(
        [(nbr, G[name_of_org][nbr]["weight"]) for nbr in G.neighbors(name_of_org)],
        key=lambda x: x[1],
        reverse=True
    )[:n_neighbors_to_display])
#EOF