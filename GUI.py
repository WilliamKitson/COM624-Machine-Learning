import streamlit as st
import utils
import data_collection_and_preprocessing
import feature_engineering
import exploratory_data_analysis
import clustering_and_grouping

st.set_page_config(page_title="4kitsw10 COM624 AE1", layout="wide")

def data_collection_page():
    if "missing_data_before" not in st.session_state:
        st.session_state.missing_data_before = None

    if "cleaned_dataset" not in st.session_state:
        st.session_state.cleaned_dataset = None

    if "missing_data_after" not in st.session_state:
        st.session_state.missing_data_after = None

    st.title("Data Collection and Pre-processing")
    st.markdown(
        "This page performs pre-processing on my dataset (CEAS_08). The cleaning process includes "
        "converting columns into more useful types, dropping columns not useful for analysis, "
        "replacing null subjects and receivers with empty strings, replace invalid dates with mean date "
        "and anonymising email addresses and links within subjects and bodies."
    )

    df = utils.load_dataset('CEAS_08.csv')
    st.dataframe(df)

    if st.button(label="Clean"):
        st.session_state.missing_data_before = utils.visualise_missing_rows(df).gcf()
        st.pyplot(st.session_state.missing_data_before)
        df = data_collection_and_preprocessing.clean_dataset(df)
        st.session_state.cleaned_dataset = df
        st.dataframe(st.session_state.cleaned_dataset)
        st.session_state.missing_data_after = utils.visualise_missing_rows(df).gcf()
        st.pyplot(st.session_state.missing_data_after)

def feature_engineering_page():
    st.title("Feature Engineering")

    if "feature_engineered_dataset" not in st.session_state:
        st.session_state.feature_engineered_dataset = None

    df = utils.load_dataset('preprocessed_dataset.csv')
    st.dataframe(df)

    if st.button(label="Engineer"):
        st.session_state.feature_engineered_dataset = feature_engineering.engineer_features(df)
        st.dataframe(st.session_state.feature_engineered_dataset)

def exploratory_data_analysis_page():
    st.title("Exploratory Data Analysis")

    if st.button(label="Analyse domains"):
        for figure in exploratory_data_analysis.analyse_domains():
            st.pyplot(figure)

def clustering_and_grouping_page():
    st.title("Clustering and Grouping")
    st.markdown(
        "This page performs clustering and grouping using K-Means clustering and DBSCAN. "
        "The optimum cluster count for K-Means is calculated using the Elbow Method, although you can define the number of clusters you want manually using the slider."
    )

    if "elbow_fig" not in st.session_state:
        st.session_state.elbow_fig = None

    if "kmeans_fig" not in st.session_state:
        st.session_state.kmeans_fig = None

    if "dbscan_fig" not in st.session_state:
        st.session_state.dbscan_fig = None

    if st.button("Run Elbow Method"):
        st.session_state.elbow_fig = clustering_and_grouping.calculate_elbow_method().gcf()

    if st.session_state.elbow_fig:
        st.pyplot(st.session_state.elbow_fig)

    kmeans_slider = st.slider("Cluster Count", 2, 20)

    if st.button("K-Means Cluster"):
        st.session_state.kmeans_fig = clustering_and_grouping.calculate_kmeans(kmeans_slider).gcf()

    if st.session_state.kmeans_fig:
        st.pyplot(st.session_state.kmeans_fig)

    if st.button("DBSCAN"):
        st.session_state.dbscan_fig = clustering_and_grouping.calculate_DBSCAN().gcf()

    if st.session_state.dbscan_fig:
        st.pyplot(st.session_state.dbscan_fig)

def privacy_preservation_page():
    st.title("Privacy Preservation")
    st.button(label="Privatise", on_click=None)

def training_page():
    st.title("Traditional Models, LSTM, and BERT")
    st.button(label="Train", on_click=None)

def prediction_page():
    st.title("Prediction")

    st.markdown(
        "Please use the below tools to run a report on simulated emails."
    )

    st.selectbox("Select Model", [
        'Random Forest',
        'Logistic Regression',
        'Naive Bayes',
        'XGBoost',
        "LSTM",
        "BERT"
    ])
    st.text_input(label="Sender", placeholder="some.dude@someplace.com")
    st.time_input(label="Time")
    st.text_input(label="Subject", placeholder="Some Subject")
    st.text_area(label="Body", placeholder="Some Body")
    st.button(label="Predict", on_click=None)

st.navigation([
    st.Page(data_collection_page, title="Data Collection and Pre-processing"),
    st.Page(feature_engineering_page, title="Feature Engineering"),
    st.Page(exploratory_data_analysis_page, title="Exploratory Data Analysis"),
    st.Page(clustering_and_grouping_page, title="Clustering and Grouping"),
    st.Page(privacy_preservation_page, title="Privacy Preservation"),
    st.Page(training_page, title = "Training"),
    st.Page(prediction_page, title = "Prediction"),
]).run()