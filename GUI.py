import streamlit as st

st.set_page_config(page_title="4kitsw10 COM624 AE1", layout="wide")
st.title("Phishing Email Detection")

st.markdown(
    "This is the GUI for my COM624 Machine Learning project."
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
st.text_input(label="Subject", placeholder="Some Subject")
st.text_area(label="Body",placeholder="Some Body")
st.button(label="Submit", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", icon=None, disabled=False, use_container_width=None, width="content")