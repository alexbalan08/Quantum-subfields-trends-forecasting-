import pandas as pd
import ast


"""
It's all exaplined on the github page what the code does step by step 
It can be found in the notebook country_contributions.

Please read that before.
"""


def load_and_prepare_data(research_path: str, patents_path: str):

    research = pd.read_csv("data/cleaned_final_data.csv")
    patents = pd.read_csv("data/patents_labeled.csv")


    research["Country"] = research["Country"].apply(ast.literal_eval)
    patents["Country"] = patents["Country"].apply(lambda x: [x] if isinstance(x, str) else [])


    research_exploded = research.explode("Country")
    patents_exploded = patents.explode("Country")


    research_exploded["Country"] = research_exploded["Country"].apply(clean_country)
    patents_exploded["Country"] = patents_exploded["Country"].apply(clean_country)


    research_counts = (
        research_exploded.groupby(["Label", "Country"])
        .size()
        .reset_index(name="Research_Count")
    )

    patent_counts = (
        patents_exploded.groupby(["Label", "Country"])
        .size()
        .reset_index(name="Patent_Count")
    )


    combined = pd.merge(research_counts, patent_counts, on=["Label", "Country"], how="outer")
    combined.fillna(0, inplace=True)


    combined["Country"] = combined["Country"].apply(normalize_country_name)

    valid_countries = set(research_exploded["Country"].unique())
    filtered_combined = combined[combined["Country"].isin(valid_countries)]


    filtered_combined["Total"] = filtered_combined["Research_Count"] + filtered_combined["Patent_Count"]

    return filtered_combined


def clean_country(name):
    name = name.strip()
    if "Wipo" in name or "PCT" in name:
        return "International"
    return name

#this is important to understand and help me to improve it please read the notebook before
def normalize_country_name(country):
    if country == "ROC":
        return "Taiwan"
    return country


def get_ordered_countries_only(filtered_combined: pd.DataFrame, label: str, top_n: int = 10):
    """
    Returns the top contributing countries for a given label, ordered by total contributions.
    Output includes only 'Label' and 'Country' columns.
    """
    filtered = filtered_combined[filtered_combined["Label"] == label]
    sorted_filtered = filtered.sort_values("Total", ascending=False)
    result = sorted_filtered[["Label", "Country", "Total"]].head(top_n).reset_index(drop=True)
    return result



