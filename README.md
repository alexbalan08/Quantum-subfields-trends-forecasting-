# Quantum Subfield Forecasting Prototype

This repository presents a complete pipeline for analyzing and forecasting the evolution of **38 quantum technology subfields** by combining data from **research publications**, **patent filings**, and **public funding programmes** (more than 70,000 records in total, 2017–2025).
The project integrates data collection, cleaning, labeling, modeling, and visualization into one workflow, and includes an interactive **Streamlit app** for exploration.

Moreover, Leo contributed with the network science part, by analyzing collaborations between different countries and institutions based on research publications (institution, country, and topical networks).

---

## Data Sources

Publications and patents were retrieved with the single keyword *quantum* rather than a fixed list of terms or patent classification codes, so that no subfield is left out in advance. Records that turned out not to be about quantum technology were removed in the labeling step.

### Research Publications
- Collected between **2017–2025**, 25,000 publications from *Scopus library*.
- Each publication labeled into one of the **38 verified quantum subfields** with **Claude 3 Haiku API**
- Invalid or uncertain classifications were excluded, so all data is ready for modelling

### Patents
- **50,000 quantum-related patents** collected from *The Lens* (aligned with 2017–2025 June).
- Data includes: title, abstract, inventor names, affiliations, and jurisdiction.
- Labeled into 38 subfields using **Claude 3 Haiku API** (title + abstract) - same labels as for the research dataset.
- Invalid or uncertain labels removed → final dataset contains **46,106 unique patents**.

### Financial Data - collected by Irene.
- **EU:** projects funded under Horizon 2020 and Horizon Europe (2017–2024) from the official *CORDIS* database, kept when the project title contains *quantum* (title, funding amount, start/end dates, participating institutions).
- **Outside the EU:** no central repository exists, so figures come from national programmes, government reports, and industry publications, converted to millions of euros and aggregated into annual estimates. UK data after Brexit are incomplete but kept.
- Funding is summed across countries into **one value per year**, used for every subfield. It therefore moves all subfields up or down together and does not explain differences between them. Because sources differ in scope and quality, funding gets only a 10% weight in the base setting.

---

## Methods I used

1. **Data Collection** → research papers, patents, and funding (2017–2025).
2. **Preprocessing & Cleaning** → remove duplicates, invalid labels, standardize affiliations/countries and author names
3. **Labeling** → auto-classification into 38 subfields, manual validation on samples.
4. **Modeling** → polynomial regression (adaptive degree 1–5) + ridge regularization (α=0.1) to avoid overfitting. We picked this based on comparison with other models as well.
   - All three indicators are min-max scaled to [0, 1]. Publications and patents each use one scaler shared across all subfields, so differences between subfields are kept; funding is scaled on its annual series.
   - Every scaler is fitted on the training years only (**2017–2023**) and applied unchanged to later years, so there is no information leakage (values from 2024 onward can exceed 1).
5. **Weighted Predictions** → combine sources with customizable weight settings:
   - Base (55% patents / 35% research / 10% funding).
   - Equal weights (33/33/33).
   - Patents + research only (50/50).
   - Custom (user-defined).

   - This allows the user to study correlation between trends score and any of the 3 data sources used and isolate if desired any source.
6. **Collaboration Networks** → from nearly 25,000 publications, institution, country, and topical networks built over the same subfields and years:
   - Institution names normalized to canonical identifiers.
   - Standard centrality measures, plus **network efficiency loss** to measure how much overall connectivity drops when a country is removed.
   - Networks exported in **GEXF** format.

---

## Streamlit Prototype

The **interactive app** allows users to:

- Forecast future growth (2026–2028) for any of the 38 subfields.
- Compare multiple subfields side by side - you can compare all of them to identify the most growing ones.
- Explore country level contributions (bar charts + maps). Which countries contributed the most in terms of reseearch/patents/investments within this topics?
- Export graphs and data as CSV or PNG.
- Get explanations of model choices (RMSE, polynomial degree, weights in the interface, good for future testing or reproducing work).

---

## Structure
In the streamlit folder all the application files are ready to be run (after installing the reqiored libraries, including streamlit)
In the rest are all the notebooks with all the steps, results and explanations are present. Those include data labeling, a frame for the predictive model with experiments including for the regularization method, etc.
The repository also contains the datasets, the institution-normalization maps, and the collaboration networks in GEXF format.
Before cecking the streamlit folder, please look over the notebooks.
The report presents the work in detail as well.


---
## Creators:
Alexandru Balan

Irene Colombo

Leo Paggen (Network Science part), canonical insitutions dataset, full networks projections
