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

def build_network(df: pd.DataFrame, map_names: Dict[str, str]) -> nx.Graph:
    inv_map: dict = build_inverse_mapping(map_names)
    df["Affiliations"] = (df["Affiliations"].fillna("").apply(lambda x: [a.strip() for a in x.split(";") if a.strip()]))
    df["Affiliations_Mapped"] = df["Affiliations"].apply(lambda lst: [inv_map.get(a, a) for a in lst])
    G = nx.Graph()
    try:
        for aff_list in df["Affiliations_Mapped"]:
            aff_list = list(set(aff_list))
            if len(aff_list) < 2:
                continue
            for u, v in combinations(aff_list, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
        print("Graph succesfully created!")
        print("Nodes:", G.number_of_nodes())
        print("Edges:", G.number_of_edges())
    except Exception as e:
        print(f"Error while building graph: {e}")
#EOF