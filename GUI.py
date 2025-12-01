import streamlit as st

st.set_page_config(page_title="4kitsw10 COM624 AE1", layout="wide")
st.title("Phishing Email Detection")

st.markdown(
    "This is the GUI for my COM624 Machine Learning project."
)

st.text_input(label="Sender", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder="some.dude@someplace.com", disabled=False, label_visibility="visible", icon=None, width="stretch")
st.text_input(label="Subject", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder="Some Subject", disabled=False, label_visibility="visible", icon=None, width="stretch")
st.text_area(label="Body", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Some Body", disabled=False, label_visibility="visible", width="stretch")