import streamlit as st
import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.pipeline import parse_transcript, compute_stats, enrich_transcript
import matplotlib.pyplot as plt
import pandas as pd
from backend import visualization as viz
from backend.agent import run_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Transcript AI Agent", layout="wide")

st.title("📝 Transcript AI Agent (MVP)")
st.write("Upload a meeting transcript JSON and analyze it!")

st.sidebar.header("Upload Transcript")
uploaded_file = st.sidebar.file_uploader("Choose a transcript JSON file", type=["json"])

if uploaded_file is not None:
    transcript_json = json.load(uploaded_file)
    df = parse_transcript(transcript_json)
    df = enrich_transcript(df)
    st.success("✅ Transcript loaded & enriched!")

    with st.expander("🔍 Transcript Preview"):
        st.dataframe(df.head(20))

    st.subheader("💬 Ask a question about the meeting")
    user_query = st.text_input("Enter your query (e.g., 'Who spoke the most?', 'Show sentiment trend')")

    if user_query:
        run_agent(user_query, df)
    else:
        st.info("⬅ Upload a transcript JSON file to get started")
        
