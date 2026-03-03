import json
import re
import unicodedata


def myhook(pairs):
        d = {}
        for k, v in pairs:
            if not isinstance(v, list):
                v = [v]
            if k in d:
                d[k].extend(v)
            else:
                d[k] = v.copy()
        return d

def order_and_save_merge_duplicates(path_to_input_map: str, path_to_output: str) -> None:
    with open(path_to_input_map, 'r') as f:
        mydata = json.load(f, object_pairs_hook=myhook)
        map_sorted = dict(sorted(mydata))
    with open(path_to_output, 'w') as f:
        json.dump(map_sorted, f)

def normalize_key(name: str) -> str:  # remove accents, lower, translate if needed
    if re.match(r'^(TU |ETH |EPFL|MIT|CERN|IBM|AWS|NEST|INFN|CNR|IIT)', name):
        return name
    name = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in name if not unicodedata.category(c) == 'Mn'])
    translations = {
        r"Universita degli Studi di ": "University of ",
        r"Universita di ": "University of ",
        r"Universite de ": "University of ",
        r"Universite d'": "University of ",
        r"Universidad de ": "University of ",
        r"Universidade de ": "University of ",
        r"Universidade Federal de ": "Federal University of ",
        r"Universite ": "University of ",
        r"Universitat ": "University of ",
        r"Universiteit ": "University of ",
        r"Universitet ": "University of ",
        r"Uniwersytet ": "University of ",
        r"Univerzita ": "University of ",
        r"The University of ": "University of ",
        r"Univ\. of ": "University of ",
        r"Univ ": "University of ",
        r"University of the ": "University of ",
        r"U\. of ": "University of ",
    }
    new_name = name
    for pattern, replacement in translations.items():
        new_name = re.sub(pattern, replacement, new_name, flags=re.IGNORECASE)
    new_name = new_name.replace(".", "").replace("-", " ")
    new_name = re.sub(r'\s+', ' ', new_name)
    return new_name.strip().lower()
#EOF