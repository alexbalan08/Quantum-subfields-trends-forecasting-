import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def prepare_data():
    """ Prepares and merges all 3 datasets togheter. 
        Normalization of patent counts + aggregating per year
    Input: Patents, research and financial data
    Output: Dataset containing all 3 different sources of data, label_counts to select degree
    
    
    """
    patents = pd.read_csv("data/patents_labeled.csv")
    research = pd.read_csv("data/cleaned_final_data.csv")
    financial = pd.read_csv("data/quantum_funding_with_all_countries.csv", sep=";")

    patents.rename(columns={"Publication Year": "Year"}, inplace=True)

    #check again github for the notebook called predcitions to see this in detail
    financial_long = financial.melt(id_vars="year", var_name="Country", value_name="Count")
    financial_long["Count"] = pd.to_numeric(financial_long["Count"], errors="coerce")
    financial_grouped = financial_long.groupby("year", as_index=False)["Count"].sum()
    financial_grouped.rename(columns={"year": "Year", "Count": "Financial"}, inplace=True)

    scaler = MinMaxScaler()
    financial_grouped["Financial_normalized"] = scaler.fit_transform(financial_grouped[["Financial"]])

    #Because we will use per year index prediction we need to group all data by year so we count 
    #each different label per each year
    #and then we apply for both research and patents
    def count_per_label_per_year(df):
        return df.groupby(["Year", "Label"]).size().reset_index(name="Count")
    
    
    patent_counts = count_per_label_per_year(patents)
    research_counts = count_per_label_per_year(research)

    #we merge patents and ficnaicial and then join fincnial data by year. 
    combined = pd.merge(patent_counts, research_counts, on=["Year", "Label"], how="outer", suffixes=("_patents", "_research"))
    combined = pd.merge(combined, financial_grouped[["Year", "Financial_normalized"]], on="Year", how="left")
    combined.fillna(0, inplace=True)

    #the final normalization for the counts
    combined[["Count_patents", "Count_research"]] = scaler.fit_transform(combined[["Count_patents", "Count_research"]])
    combined.rename(columns={"Financial_normalized": "Financial"}, inplace=True)

    #we start with this basic but we can oveeride this later from the GUI !!!

    
    #!!!
    combined["WeightedScore"] = (
        0.55 * combined["Count_patents"] +
        0.35 * combined["Count_research"] +
        0.10 * combined["Financial"]
    )
    #!!!

    
    combined = combined[~combined["Label"].isin(["error", "invalid_label"])].copy()

    #we calculate how many patents + research papers are for each topic in particular
    #then we create another collumn with counts that we use for our final data 
    #this will also be used to determine the degree of the model depending on how much data is there
    patent_counts_per_label = patents["Label"].value_counts().reset_index()
    patent_counts_per_label.columns = ["Label", "Patent_Count"]

    #this exaclty the same 
    research_counts_per_label = research["Label"].value_counts().reset_index()
    research_counts_per_label.columns = ["Label", "Research_Count"]

    label_counts = pd.merge(patent_counts_per_label, research_counts_per_label, on="Label", how="outer").fillna(0)
    label_counts["Patent_Count"] = label_counts["Patent_Count"].astype(int)
    label_counts["Research_Count"] = label_counts["Research_Count"].astype(int)
    label_counts["Total_Count"] = label_counts["Patent_Count"] + label_counts["Research_Count"]
    label_counts = label_counts.sort_values("Total_Count", ascending=False).reset_index(drop=True)
    
    #combined is what we need to use from now as the dataset, label_counts is jut for degree selectiopn
    return combined, label_counts


def determine_degree(total_count):
    """Estabblishes the degree based on how much data is avilable per a topic

       Input: total_counts per topic combined with research papers and patents 
       Output: the polynomial degree ranginf from 1 to 5 
    """
    if total_count >= 8000:
        return 5
    elif total_count >= 2000:
        return 3
    elif total_count >= 1000:
        return 2
    else:
        return 1


def run_forecast(label_name, w_patents, w_research, w_financial, alpha, combined, label_counts, future_years):

    row = label_counts[label_counts["Label"] == label_name]
    if row.empty:
        st.error(f"Label not found: {label_name}")
        return

    total_count = row["Total_Count"].values[0]
    degree = determine_degree(total_count)

    # Filter data for selected label
    df = combined[combined["Label"] == label_name][["Year", "Count_patents", "Count_research", "Financial"]].copy()
    # Always use fixed weights for past data (for comparability)
    df["WeightedScore"] = (
    0.55 * df["Count_patents"] +
    0.35 * df["Count_research"] +
    0.10 * df["Financial"]
)

#we use this custom scores we pass in in the app.py class from the user interface
    df["CustomScore"] = (
    w_patents * df["Count_patents"] +
    w_research * df["Count_research"] +
    w_financial * df["Financial"]
)
    df = df[(df["Year"] >= 2017) & (df["Year"] <= 2024)]

    #we use only 2024 for validation rest for training initally
    #2024 has the most data

    train_df = df[df["Year"] <= 2023]
    val_df = df[df["Year"] == 2024]

    X_train = train_df["Year"].values.reshape(-1, 1)
    y_train = train_df["CustomScore"].values
    X_val = val_df["Year"].values.reshape(-1, 1)
    y_val = val_df["WeightedScore"].values

    poly = PolynomialFeatures(degree= degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = Ridge(alpha=alpha)

    #!!IMPORTANT!! If you want to weight recent years more:
    # year_weights = np.linspace(1.0, 2.0, len(y_train))
    # model.fit(X_train_poly, y_train, sample_weight=year_weights)

    model.fit(X_train_poly, y_train)
    val_pred = model.predict(X_val_poly)

    rmse = mean_squared_error(y_val, val_pred) ** 0.5
    st.markdown(f"**Degree = {degree}**, Validation RMSE: `{rmse:.4f}`")


    #now we retrain the model on all data because having the last 2 years in the training is very important
    #it has the most growth so if we dont use them for trianing the model wont be relaible\

    #X_full = df["Year"].values.reshape(-1, 1)
    #y_model = df["CustomScore"].values
    #model.fit(poly.fit_transform(X_full), y_model)

    ##we uncomment this if we want to reuse all data for traiing 

    y_plot = df["WeightedScore"].values
    y_full   = y_plot 

    # X_future = np.arange(2025, 2029).reshape(-1, 1)
    # y_future = model.predict(poly.transform(X_future))
    X_future = np.array(future_years).reshape(-1, 1)
    y_future = model.predict(poly.transform(X_future))

    
    all_years = np.concatenate([df["Year"].values, X_future.flatten()])
   
    actual_years = df["Year"].values
    predicted_years = X_future.flatten()

    #baseline is just the last year available so 2024 we get with index -1
    baseline = y_full[-1]
    growth = [((score - baseline) / baseline) * 100 if baseline > 0 else 0 for score in y_future]
    growth_scores = [baseline * (1 + g / 100) for g in growth]
 

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')


    ax.bar(actual_years, y_full, label="Actual (2017–2024)", color="steelblue")
    #ax.bar(predicted_years, growth_scores, label="Predicted (2025–2028)",       color="orange")
    year_range_str = f"{min(future_years)}–{max(future_years)}" if len(future_years) > 1 else f"{future_years[0]}"
    ax.bar(predicted_years, growth_scores, label=f"Predicted ({year_range_str})", color="orange")



    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years.astype(int))
    ax.xaxis.set_major_locator(ticker.FixedLocator(all_years))

    ax.set_title(f"Forecast for {label_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(False)

    fig.tight_layout()
    st.pyplot(fig)




    st.markdown("### 📈 Predicted Growth in percentages (vs. 2024)")

    fig_growth, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["steelblue"] * len(growth)
    bars = ax.barh(predicted_years.astype(str), growth, color=colors, edgecolor="black", height=0.6)

    ax.set_xlabel("Growth (%)", fontsize=10)
    ax.set_xlim(min(-10, min(growth) - 5), max(growth) + 5)
    ax.set_title("Forecasted Growth Compared to 2024", fontsize=12, weight="bold")
    ax.axvline(0, color="gray", linewidth=1.2, linestyle="--")

    #final section with percentages growth
    #we can maybe customize this later
    for bar, g in zip(bars, growth):
       ax.text(bar.get_width() + (2 if g >= 0 else -2), bar.get_y() + bar.get_height()/2,
            f"{g:.1f}%", va='center', ha='left' if g >= 0 else 'right', fontsize=9, weight="bold")

    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    
    st.pyplot(fig_growth)

    
    # Prepare data to download
    download_df = pd.DataFrame({
        "Year": X_future.flatten(),
        "Predicted Score": y_future,
        "Topic": label_name
    })

    csv = download_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="💾 Download predictions as CSV",
        data=csv,
        file_name=f"{label_name}_forecast.csv",
        mime='text/csv'
    )

    import io

    # Save plot to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    st.download_button(
        label="🖼️ Download chart as PNG",
        data=buf,
        file_name=f"{label_name}_forecast.png",
        mime="image/png"
    )


    return {
    "label": label_name,
    "years": list(df["Year"].values) + list(predicted_years),
    "scores": list(y_full) + list(growth_scores),
    "is_predicted": [False] * len(y_full) + [True] * len(growth_scores)
}






