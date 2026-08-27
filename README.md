# Quantum Subfield Forecasting Prototype

This repository presents a complete pipeline for analyzing and forecasting the evolution of **38 quantum technology subfields** by combining data from **research publications**, **patent filings**, and **public/private funding sources**.
The project integrates data collection, cleaning, labeling, modeling, and visualization into one workflow, and includes an interactive **Streamlit app** for exploration.

Moreover, Leo contributed with the network science part, by analyzing collaborations between different countries and institutions based on patents fillings.

---

## Data Sources

### Research Publications
- Collected between **2017–2024**, 25,000 publications from *Scopus library*.
- Each publication labeled into one of the **38 verified quantum subfields** with **Claude 3 Haiku API**
- Invalid or uncertain classifications were excluded, so all data is ready for modelling

### Patents
- **50,000 quantum-related patents** collected from *The Lens* (aligned with 2017–2025 June).
- Data includes: title, abstract, inventor names, affiliations, and jurisdiction.
- Labeled into 38 subfields using **Claude 3 Haiku API** (title + abstract) - same labels as for the research dataset.
- Invalid or uncertain labels removed → final dataset contains **46,106 unique patents**.

### Financial Data - collected by Irene.

---

## Methods I used

1. **Data Collection** → research papers, patents, and funding (2017–2025).
2. **Preprocessing & Cleaning** → remove duplicates, invalid labels, standardize affiliations/countries and author names
3. **Labeling** → auto-classification into 38 subfields, manual validation on samples.
4. **Modeling** → polynomial regression (adaptive degree 1–5) + ridge regularization (α=0.1) to avoid overfitting. We picked this based on comparison with other models as well.
5. **Weighted Predictions** → combine sources with customizable weight settings:
   - Base (55% patents / 35% research / 10% funding).
   - Equal weights (33/33/33).
   - Patents + research only (50/50).
   - Custom (user-defined).

   - This allows the user to study correlation between trends score and any of the 3 data sources used and isolate if desired any source.

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
Before cecking the streamlit folder, please look over the notebooks.
The report presents the work in detail as well.



---
## Creators:
Alexandru Balan

Irene Colombo

Leo Paggen (Network Science part), canonical insitutions dataset, full networks projections
