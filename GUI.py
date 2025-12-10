import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import utils
import data_collection_and_preprocessing
import feature_engineering
import exploratory_data_analysis
import clustering_and_grouping
import privacy_preservation
import traditional_models
from datetime import datetime

# Please find and download the CEAS_08 dataset at: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
# This dataset MUST be pasted into the datasets directory

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

    st.markdown(
        "This page performs feature engineering, the process of extracting new features from the cleaned "
        "dataset. Please run feature engineering and compare the cleaned dataset to the produced feature "
        "engineered dataset"
    )

    if "feature_engineered_dataset" not in st.session_state:
        st.session_state.feature_engineered_dataset = None

    df = utils.load_dataset('preprocessed_dataset.csv')
    st.dataframe(df)

    if st.button(label="Engineer"):
        st.session_state.feature_engineered_dataset = feature_engineering.engineer_features(df)
        st.dataframe(st.session_state.feature_engineered_dataset)

def exploratory_data_analysis_page():
    st.title("Exploratory Data Analysis")

    st.markdown(
        "This page performs Exploratory Data Analysis, please click on the buttons below to visualise "
        "the corresponding EDA insights."
    )

    if st.button(label="Analyse domains"):
        for figure in exploratory_data_analysis.analyse_domains():
            st.pyplot(figure)

    if st.button(label="Box Plot Body Length"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('body_length'))

    if st.button(label="Box Plot Subject Length"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('subject_length'))

    if st.button(label="Box Plot Total Word Count"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('total_word_count'))

    if st.button(label="Box Plot Link Count"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('link_count'))

    if st.button(label="Box Plot Misspellings"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('misspellings'))

    if st.button(label="Box Plot Correct Spellings"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('correct_spellings'))

    if st.button(label="Box Plot Correct Spellings Scaled"):
        st.pyplot(exploratory_data_analysis.boxplot_columns('correct_spellings_scaled'))

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
    if "privacy_preserved_dataset" not in st.session_state:
        st.session_state.privacy_preserved_dataset = None

    st.title("Privacy Preservation")

    st.markdown(
        "This page performs privacy preservation to the feature engineered dataset using differential "
        "privacy. You can set the differential privacy epsilon using the slider. To apply these changes "
        "to the models during training, you must save the new privacy preserved dataset using the button."
    )

    df = utils.load_dataset('feature_engineered_dataset.csv')
    st.dataframe(df)

    differential_privacy_epsilon_slider = st.slider("Differential Privacy Epsilon", 0, 100)

    if st.button("Apply Differential Privacy"):
        st.session_state.privacy_preserved_dataset = privacy_preservation.differential_privacy(
            df,
            differential_privacy_epsilon_slider
        )

    if st.button("Dont Apply Differential Privacy"):
        st.session_state.privacy_preserved_dataset = privacy_preservation.no_differential_privacy(df)

    st.dataframe(st.session_state.privacy_preserved_dataset)

    if st.button("Save Privacy Preserved Dataset"):
        utils.save_dataset(st.session_state.privacy_preserved_dataset, 'privacy_preserved_dataset.csv')

def training_page():
    st.title("Traditional Models, LSTM, and BERT")

    if st.button("Train Random Forest"):
        output = traditional_models.train_model(RandomForestClassifier(), 'Random Forest')
        st.pyplot(output['visualisation'])

    if st.button("Train Logistic Regression"):
        output = traditional_models.train_model(LogisticRegression(), 'Logistic Regression')
        st.pyplot(output['visualisation'])

    if st.button("Train Naive Bayes"):
        output = traditional_models.train_model(GaussianNB(), 'Naive Bayes')
        st.pyplot(output['visualisation'])

    if st.button("Train XGBoost"):
        output = traditional_models.train_model(XGBClassifier(), 'XGBoost')
        st.pyplot(output['visualisation'])

def prediction_page():
    st.title("Prediction")

    st.markdown(
        "Please use the below tools to run a report on simulated emails."
    )

    select_box = st.selectbox("Select Model", [
        'Random Forest',
        'Logistic Regression',
        'Naive Bayes',
        'XGBoost',
        "LSTM",
        "BERT"
    ])

    sender = st.text_input(label="Sender", placeholder="some.dude@someplace.com")
    receiver = st.text_input(label="receiver", placeholder="another.dude@someplace.com")
    date = st.date_input(label="Date")
    time = st.time_input(label="Time")
    subject = st.text_input(label="Subject", placeholder="Some Subject")
    body = st.text_area(label="Body", placeholder="Some Body")

    if st.button(label="Predict"):
        datetime_combined = datetime.combine(date, time)

        df = pd.DataFrame({
            'sender': [sender],
            'receiver': [receiver],
            'date': [datetime_combined],
            'subject': [subject],
            'body': [body],
            'label': [-1]
        })

        st.subheader("Feature Engineered Dataset")
        df = feature_engineering.engineer_features(df)
        st.session_state.prediction_feature_engineered_dataset = df
        st.dataframe(st.session_state.prediction_feature_engineered_dataset)

        st.subheader("Privacy Preserved Dataset")
        df = privacy_preservation.differential_privacy(df, 1)
        st.session_state.prediction_privacy_preserved_dataset = df
        st.dataframe(st.session_state.prediction_privacy_preserved_dataset)

        model = None

        if select_box == 'Random Forest':
            model = utils.load_model('Random Forest')

        if select_box == 'Logistic Regression':
            model = utils.load_model('Logistic Regression')

        if select_box == 'Naive Bayes':
            model = utils.load_model('Naive Bayes')

        if select_box == 'XGBoost':
            model = utils.load_model('XGBoost')

        prediction = 'Phishing'

        if model is not None:
            features = df.drop(columns=['label'])

            if model.predict(features)[0] == 0:
                prediction = 'Safe'

            st.subheader("Prediction")
            st.write(prediction)

st.navigation([
    st.Page(data_collection_page, title="Data Collection and Pre-processing"),
    st.Page(feature_engineering_page, title="Feature Engineering"),
    st.Page(exploratory_data_analysis_page, title="Exploratory Data Analysis"),
    st.Page(clustering_and_grouping_page, title="Clustering and Grouping"),
    st.Page(privacy_preservation_page, title="Privacy Preservation"),
    st.Page(training_page, title = "Training"),
    st.Page(prediction_page, title = "Prediction"),
]).run()