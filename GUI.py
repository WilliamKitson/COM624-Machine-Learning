import streamlit as st
from clustering_and_grouping import calculate_elbow_method

st.set_page_config(page_title="4kitsw10 COM624 AE1", layout="wide")

def data_collection_page():
    st.title("Data Collection and Pre-processing")
    st.button(label="Clean", on_click=None)

def feature_engineering_page():
    st.title("Feature Engineering")
    st.button(label="Engineer", on_click=None)

def exploratory_data_analysis_page():
    st.title("Exploratory Data Analysis")
    st.button(label="Analyse", on_click=None)

def clustering_and_grouping_page():
    st.title("Clustering and Grouping")
    st.markdown(
        "This page performs clustering and grouping using K-Means clustering and DBSCAN. "
        "The optimum cluster count for K-Means is calculated using the Elbow Method, although you can define the number of clusters you want manually using the slider."
    )

    elbow_placeholder = st.empty()

    if st.button(label="Run Elbow Method"):
        fig = calculate_elbow_method().gcf()
        elbow_placeholder.pyplot(fig)

    st.slider(label="Cluster Count", min_value=0, max_value=20)

    st.button(label="K-Means Cluster", on_click=None)

    st.button(label="DBSCAN", on_click=None)

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