import streamlit as st
import pycountry
import plotly.express as px
import pandas as pd
from Backend import run_forecast, determine_degree, prepare_data
from Country import load_and_prepare_data, get_ordered_countries_only


st.set_option('client.showErrorDetails', True)

@st.cache_data
def load_data():
    return prepare_data()

@st.cache_data
def load_country_data():
    return load_and_prepare_data("cleaned_final_data.csv", "patents_labeled.csv")

filtered_combined = load_country_data()
combined, label_counts = load_data()



st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("📚 Go to section:", ["📈 Forecasting", "🌍 Country Insights", "ℹ️ About"])

valid_labels = label_counts[~label_counts["Label"].isin(["error", "invalid_label"])]["Label"]
label = st.sidebar.selectbox("📌 Quantum topic", sorted(valid_labels))



if page == "📈 Forecasting":
    st.title("🔮 Quantum Subfield Growth Forecast")

    with st.expander("ℹ️ What does this dashboard show and how it works?", expanded=False):
     st.markdown("""
    This tool forecasts future growth of selected **quantum subfields** using data from:
    
    - 🧠 **Research publications**
    - 💡 **Patent filings**
    - 💰 **Financial investments**

    The model applies:
    
    - 🔁 **Polynomial regression**, with the degree chosen adaptively
    - 📉 **Ridge regularization** to prevent overfitting, Alpha parameter being preset
    - 🧮 A **weighted score** per year, based on your selected strategy
    
    ---
    ### 📊 What you can do:
    - Choose a **quantum topic**
    - Set **custom or preset weights**
    - Run a forecast for future trends
    - Explore the **top contributing countries** to each subfield selected
    
    You can tweak everything and compare the influence of different data sources by using custom and setting any of the 3 variables to 0.
    """)

    st.markdown("---")  
    st.markdown("### 🎛️ Select Weighting Strategy")

    weight_option = st.selectbox(
        "Choose a preset or use custom weights:",
        (
            "Base (Recommended) — 55% Patents / 35% Research / 10% Financial",
            "Equal Weights — 33% Patents / 33% Research / 33% Financial",
            "Only Research & Patents — 50% / 50%, No Financial",
            "Custom"
        )
    )

    with st.expander("ℹ️ How do the weights and prediction score work?"):
     st.markdown("""
    The model calculates a **Weighted Score** for each topic and year based on a combination of:

    - 🧠 **Number of Patents from subfield** — an indicator of technological innovation.
    - 📚 **Number of Research Papers from subfield** — reflect academic progress.
    - 💰 **Financial Investments** — signal market confidence.

    The score is calculated as a **weighted sum** of these three components. You can choose:

    - **Preset strategies** like ***Base*** (55%/35%/10%) or “Equal Weights”.
    - Or define your **custom weights** to select what matters most to your goals.

    ---
    Some recommendations:
     
    - Patents data includes 50k entires and it represents a larger weight for the **Base** setting. 
    - Financial data was limited so for this reason a weight of only 10% is **recommended**. 
    - For example, **Quantum Cryptography** may be heavy in research but light in commercialization, while **Quantum Hardware** might attract significant investment: for those cases, custom weights can be picked.
    """)

    if weight_option == "Base (Recommended) — 55% Patents / 35% Research / 10% Financial":
        w_patents, w_research, w_financial = 0.55, 0.35, 0.10
    elif weight_option == "Equal Weights — 33% Patents / 33% Research / 33% Financial":
        w_patents, w_research, w_financial = 0.33, 0.33, 0.34
    elif weight_option == "Only Research & Patents — 50% / 50%, No Financial":
        w_patents, w_research, w_financial = 0.5, 0.5, 0.0
    else:
        st.markdown("#### 🔧 Customize Your Own Weights")
        w_patents = st.slider("Patent weight", 0.0, 1.0, 0.55)
        w_research = st.slider("Research weight", 0.0, 1.0, 0.35)
        w_financial = st.slider("Financial weight", 0.0, 1.0, 0.10)
        total = w_patents + w_research + w_financial
        if total > 0:
            w_patents /= total
            w_research /= total
            w_financial /= total

    alpha = 0.1  # regularization

    if st.button("🚀 Run Forecast"):
        run_forecast(
            label_name=label,
            w_patents=w_patents,
            w_research=w_research,
            w_financial=w_financial,
            alpha=alpha,
            combined=combined,
            label_counts=label_counts
        )



elif page == "🌍 Country Insights":
    st.title("🌍 Top Contributing Countries")
    top_n = st.slider("Number of countries to display", 5, 20, 10)
    top_countries_df = get_ordered_countries_only(filtered_combined, label, top_n)

    # Table
    top_countries_df_display = top_countries_df.copy()
    top_countries_df_display.index = top_countries_df_display.index + 1
    top_countries_df_display.index.name = "Rank"

    st.dataframe(top_countries_df_display)

    # Bar chart
    st.bar_chart(data=top_countries_df.set_index("Country")["Total"])

    # Map
    with st.expander("🗺️ View Country Contributions on Map"):
        all_countries = sorted({country.name for country in pycountry.countries})
        world_df = pd.DataFrame({"country": all_countries})

        map_df = top_countries_df.copy().rename(columns={"Country": "country", "Total": "contribution"})
        merged_map = world_df.merge(map_df, on="country", how="left")
        merged_map["contribution"] = merged_map["contribution"].fillna(0)

        merged_map["hover"] = merged_map.apply(
            lambda row: f"{row['country']}: {int(row['contribution'])}" if row["contribution"] > 0 else "",
            axis=1
        )
        merged_map["status"] = merged_map["contribution"].apply(
            lambda x: "Contributing" if x > 0 else "No contribution"
        )

        color_map = {
            "No contribution": "#E0E0E0",
            "Contributing": "#1f77b4"
        }

        fig = px.choropleth(
            merged_map,
            locations="country",
            locationmode="country names",
            color="status",
            hover_name="hover",
            color_discrete_map=color_map,
            title=f"🌍 Contributions by Country – {label}",
            projection="natural earth"
        )

        fig.update_geos(
            showcountries=True,
            showcoastlines=True,
            showframe=False,
            fitbounds="locations"
        )
        fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)



elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown("""
    This web-app forecasts quantum subfield growth using multiple types of data:
    
    - 🧠 Research data  
    - 💡 Patent data  
    - 💰 Financial trends  
                
    See each individual section information for more details.

    The scoring model adapts the polynomial degree based on data volume and uses ridge regression for better generalization.

    ---
    **Built with:** Python, Streamlit, Plotly  
                
    **Author:** [Alex Balan](https://github.com/alexbalan08)  
                
    **Patents and research datasets creator:** [Alex Balan](https://github.com/alexbalan08)
                
    **Financial dataset creator:** [Irene Colombo](https://github.com/irenecolomboo)
                
    **GitHub:** [later](https://github.com/your-repo)

    Feel free to open a pull request if you'd like to contribute!
    """)
