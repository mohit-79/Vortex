import json
import pandas as pd
from typing import Dict, Any, List
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from spacy.lang.en import English

nlp=spacy.load("en_core_web_sm")

def extract_relations_spacy(text):
    doc = nlp(text)
    relations = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = []
                objects = []
                
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subjects.append(child.text)
                    if child.dep_ in ("dobj", "pobj"):
                        objects.append(child.text)
                
                for subj in subjects:
                    for obj in objects:
                        relations.append((subj, token.lemma_, obj))
    return relations

def extract_keywords(text: str, top_n: int = 5):
    doc = nlp(text.lower())
    keywords = [chunk.text for chunk in doc.noun_chunks]
    # named_entities = [ent.text for ent in doc.ents]
    
    named_entities=[(ent.text, ent.label_) for ent in doc.ents]
    return {
        "keywords": keywords[:top_n],
        "named_entities": named_entities
    }

def analyze_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def enrich_transcript(df):
    df["keywords"] = df["text"].apply(lambda t: extract_keywords(t, top_n=5)["keywords"])
    df["Named Entity Recognitation"]=df["text"].apply(lambda t: extract_keywords(t, top_n=5)["named_entities"])
    df["relations"] = df["text"].apply(extract_relations_spacy)
    df["sentiment"] = df["text"].apply(analyze_sentiment)
    # df["topics"] = extract_topics_per_doc(df["text"].tolist(), max_features=5)
    return df

def compute_sentiment_trends(df):
    return df.groupby("speaker").agg(
        avg_sentiment=("sentiment", "mean"),
        turns=("sentiment", "count")
    ).reset_index()

def load_transcript(file) -> Dict[str, Any]:
    """Load transcript JSON into a dict"""
    return json.load(file)


def parse_transcript(transcript_json: Dict[str, Any]) -> pd.DataFrame:
    """Convert transcript JSON into a DataFrame"""
    records: List[Dict[str, Any]] = []

    for entry in transcript_json:
        start = entry["start_timestamp"]
        end = entry["end_timestamp"]
        duration = end - start
        word_count = len(entry["text"].split())

        records.append({
            "speaker": entry["name"],
            "text": entry["text"],
            "start_time": start,
            "end_time": end,
            "duration": duration,
            "word_count": word_count,
        })

    return pd.DataFrame(records)


# def timestamp_to_seconds(ts: str) -> int:
#     """Convert HH:MM:SS string into seconds"""
#     h, m, s = map(int, ts.split(":"))
#     return h * 3600 + m * 60 + s


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic per-speaker statistics"""
    stats = df.groupby("speaker").agg(
        total_words=("word_count", "sum"),
        total_duration=("duration", "sum"),
        turns=("text", "count")
    ).reset_index()
    return stats
