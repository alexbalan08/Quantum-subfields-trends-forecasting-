import pandas as pd
import networkx as nx
from typing import Dict
from itertools import combinations
import ast
import json


# constants
PATH_TO_CANON_MAP = ""
PATH_TO_DF = ""  # in .csv!!!!!!!
PATH_TO_COUNTRY_MAP = ""


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

def build_G_inst(df: pd.DataFrame, country_cleaning_map: dict[str, str]) -> nx.Graph:
    G = nx.Graph()
    try:
        for _, row in df.iterrows():
            aff_list = row["Affiliations_Mapped"]
            raw_list = row["Affiliations"]  # original raw strings
            unique_pairs = list(set(zip(aff_list, raw_list)))
            unique_affs = list(set(aff_list))
            N = len(unique_affs)
            for canon, raw in unique_pairs:
                country_raw = raw.split(",")[-1].strip()
                country = country_cleaning_map['canon_map'].get(country_raw, country_raw)
                if canon not in G:
                    G.add_node(canon, country=country)
            if N >= 2:
                w = newman_weight(N)
                for u, v in combinations(unique_affs, 2):
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += w
                    else:
                        G.add_edge(u, v, weight=w)
        print("Graph successfully created!")
        print("Nodes:", G.number_of_nodes())
        print("Edges:", G.number_of_edges())
    except Exception as e:
        print(f"Error while building graph: {e}")
    return G

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

def build_G_country(df: pd.DataFrame, country_cleaning_map: dict[str, str]) -> nx.Graph:
    """
    Builds a country-to-country collaboration graph.
    All variations (e.g., 'Macau', 'Puerto Rico') are funneled into
    Canon Country Nodes via the country_cleaning_map.
    Node attributes include lat/lon from country_cleaning_map['coords'].
    """
    df['Country_Parsed'] = df['Country'].fillna("[]").apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [x])
    )
    coords = country_cleaning_map.get('coords', {})
    canon_map = country_cleaning_map.get('canon_map', {})
    EXCLUDE = {"Unknown/Multi-Affiliation", "Unknown"}
    G = nx.Graph()
    for country_list in df["Country_Parsed"]:
        cleaned_list = list(set(
            canon_map.get(c.strip(), c.strip())
            for c in country_list
            if c and canon_map.get(c.strip(), c.strip()) not in EXCLUDE
        ))
        N = len(cleaned_list)
        if N < 1:
            continue
        for country in cleaned_list:
            if country not in G:
                c = coords.get(country)
                G.add_node(country, lat=c[0], lon=c[1])
        if N >= 2:
            w = newman_weight(N)
            for u, v in combinations(cleaned_list, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    return G

def split_map_country(input_map: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in input_map.items():
        for org in v:  # recall v is a list of strings
            country: str = org.split(",")[-1].strip()  # country always present as the last element
            new_key = k + " (" + country + ")"
            if not out.get(new_key):
                out[new_key] = [org]
            else:
                out[new_key].append(org)
    return out

def clean_country_list(c_list: list[str], country_cleaning_map: dict[str, str]):
    return list(set(country_cleaning_map.get(c, c) for c in c_list))

def get_neighbors(name_of_org: str, G: nx.Graph, n_neighbors_to_display: int) -> None:  # it just prints
    # EXAMPLE USAGE: name_of_org = "ETH Zurich" // will try later on to have a system to recognize incorrect spellings
    print(sorted(
        [(nbr, G[name_of_org][nbr]["weight"]) for nbr in G.neighbors(name_of_org)],
        key=lambda x: x[1],
        reverse=True
    )[:n_neighbors_to_display])

def top_international_collabs(G_inst, country, n=10):
    results = []
    for u, v, d in G_inst.edges(data=True):
        cu = G_inst.nodes[u].get("country")
        cv = G_inst.nodes[v].get("country")
        if cu != cv and (cu == country or cv == country):
            partner_inst = v if cu == country else u
            partner_country = cv if cu == country else cu
            results.append((u if cu == country else v, partner_inst, partner_country, d["weight"]))
    return sorted(results, key=lambda x: x[3], reverse=True)[:n]
#EOF