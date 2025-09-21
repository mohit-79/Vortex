import re
import pandas as pd
import plotly.express as px
from textwrap import shorten
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.figure_factory as ff
import seaborn as sns
import networkx as nx

def speaker_contribution_bar(df: pd.DataFrame, by: str = "word_count"):
    import pandas as pd
    agg = df.groupby("speaker", as_index=False)[by].sum().sort_values(by, ascending=False)
    title = "Total Words Spoken" if by == "word_count" else "Total Talk Time (seconds)"
    fig = px.bar(agg, x="speaker", y=by, title=title)
    return fig, agg


def speaker_contribution_pie(df: pd.DataFrame):
    agg = df.groupby("speaker", as_index=False)["word_count"].sum()
    fig = px.pie(agg, names="speaker", values="word_count", title="Speaker % Contribution")
    return fig, agg


def keyword_frequency_chart(df: pd.DataFrame, keywords=None, top_n: int = 10):
    texts = df["text"].str.lower().fillna("")
    if keywords:
        keywords = [kw.lower() for kw in keywords]
        results = []
        for kw in keywords:
            mask = texts.str.contains(rf"\b{re.escape(kw)}\b")
            count = mask.sum()
            results.append({"keyword": kw, "count": int(count)})
        agg = pd.DataFrame(results)
    else:
        all_words = " ".join(texts).split()
        counts = pd.Series(all_words).value_counts().head(top_n).reset_index()
        counts.columns = ["keyword", "count"]
        agg = counts

    fig = px.bar(agg, x="keyword", y="count", title="Keyword Frequency")
    return fig, agg


def keyword_mentions_table(df: pd.DataFrame, keyword: str):
    kw = keyword.lower()
    mask = df["text"].str.lower().str.contains(rf"\b{re.escape(kw)}\b")
    subset = df[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["speaker", "start_time", "snippet"])
    subset["snippet"] = subset["text"].apply(lambda t: shorten(t, width=120, placeholder="..."))
    return subset[["speaker", "start_time", "snippet"]]


def sentiment_trend_line(df: pd.DataFrame):
    df_sorted = df.sort_values("start_time").copy()
    fig = px.line(df_sorted, x="start_time", y="sentiment", color="speaker",
                  title="Sentiment Trend Over Time", markers=True)
    return fig, df_sorted


def conversation_flow_graph(df: pd.DataFrame):
    df_sorted = df.sort_values("start_time").reset_index(drop=True)
    transitions = {}
    prev = None
    for _, row in df_sorted.iterrows():
        cur = row["speaker"]
        if prev:
            key = (prev, cur)
            transitions[key] = transitions.get(key, 0) + 1
        prev = cur

    lines = ["digraph G {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    for (a, b), count in transitions.items():
        lines.append(f'  "{a}" -> "{b}" [label="{count}"];')
    lines.append("}")
    dot = "\n".join(lines)
    return dot, transitions

# def wordcloud_chart(df):
#     """Generate word cloud of most common words"""
#     from wordcloud import WordCloud
#     import matplotlib.pyplot as plt
    
#     text = " ".join(df["text"].dropna().str.lower())
#     wc = WordCloud(width=800, height=400, background_color="white").generate(text)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wc, interpolation="bilinear")
#     ax.axis("off")
#     return fig

def keyword_timeline_chart(df, keyword):
    """Timeline of keyword mentions across meeting"""
    mask = df["text"].str.lower().str.contains(keyword.lower())
    subset = df[mask].copy()
    fig = px.scatter(subset, x="start_time", y="speaker", text="text",
                     title=f"Mentions of '{keyword}' over time")
    return fig

def speaker_turns_bar(df):
    """Number of speaking turns per speaker"""
    turns = df["speaker"].value_counts().reset_index()
    turns.columns = ["speaker", "turns"]
    fig = px.bar(turns, x="speaker", y="turns", title="Number of Speaking Turns")
    return fig



def conversation_timeline_chart(df: pd.DataFrame):
    """Gantt-style chart showing when each speaker talked"""
    df_sorted = df.sort_values("start_time").copy()
    tasks = []
    for _, row in df_sorted.iterrows():
        tasks.append(dict(
            Task=row["speaker"],
            Start=row["start_time"],
            Finish=row["end_time"],
            Description=row["text"]
        ))
    fig = ff.create_gantt(tasks, index_col="Task", show_colorbar=True,
                          group_tasks=True, showgrid_x=True, showgrid_y=True)
    return fig, df_sorted


# def wordcloud_per_speaker(df: pd.DataFrame):
#     """Generate word cloud images per speaker"""
#     clouds = {}
#     for speaker, group in df.groupby("speaker"):
#         text = " ".join(group["text"].dropna().astype(str))
#         if text.strip():
#             wc = WordCloud(width=400, height=300,
#                            background_color="white",
#                            colormap="tab10").generate(text)
#             clouds[speaker] = wc
#     return clouds
# def wordcloud_per_speaker(df: pd.DataFrame):
#     """Generate word cloud matplotlib figures per speaker"""
#     figures = {}
#     for speaker, group in df.groupby("speaker"):
#         text = " ".join(group["text"].dropna().astype(str))
#         if text.strip():
#             wc = WordCloud(width=400, height=300,
#                            background_color="white",
#                            colormap="tab10").generate(text)
#             fig, ax = plt.subplots(figsize=(5, 4))
#             ax.imshow(wc, interpolation="bilinear")
#             ax.set_title(f"Word Cloud: {speaker}")
#             ax.axis("off")
#             figures[speaker] = fig
#     return figures  # dict of matplotlib figures



def action_items_table(df: pd.DataFrame):
    """
    Extracts potential action items (simple heuristic).
    Looks for phrases like 'we should', 'let’s', 'need to', 'decide'.
    """
    patterns = r"(we should|let's|need to|decide|plan to|action item)"
    mask = df["text"].str.lower().str.contains(patterns, regex=True, na=False)
    subset = df[mask].copy()
    subset["snippet"] = subset["text"].apply(lambda t: shorten(t, width=120, placeholder="..."))
    return subset[["speaker", "start_time", "snippet"]]


# def sentiment_heatmap(df: pd.DataFrame, bins: int = 10):
#     """Heatmap of sentiment over time bins per speaker"""
#     df_sorted = df.sort_values("start_time").copy()
#     df_sorted["time_bin"] = pd.qcut(df_sorted["start_time"], bins, duplicates="drop")
#     pivot = df_sorted.groupby(["speaker", "time_bin"])["sentiment"].mean().unstack(fill_value=0)

#     plt.figure(figsize=(10, 6))
#     sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=False)
#     plt.title("Sentiment Heatmap (Speaker × Time)")
#     fig = plt.gcf()
#     return fig, pivot


def speaker_network_graph(df: pd.DataFrame):
    """Speaker interaction network using NetworkX"""
    df_sorted = df.sort_values("start_time").reset_index(drop=True)
    transitions = {}
    prev = None
    for _, row in df_sorted.iterrows():
        cur = row["speaker"]
        if prev:
            key = (prev, cur)
            transitions[key] = transitions.get(key, 0) + 1
        prev = cur

    G = nx.DiGraph()
    for (a, b), w in transitions.items():
        G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="skyblue")
    nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20,
                                   width=[d["weight"] for (_, _, d) in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)})
    plt.title("Speaker Interaction Network")
    fig = plt.gcf()
    return fig, transitions

def speaking_time_bar(df: pd.DataFrame):
    """
    Bar chart of total speaking time (in seconds) per speaker.
    Requires 'end_time' and 'start_time' columns.
    """
    df = df.copy()
    df["duration"] = df["end_time"] - df["start_time"]
    agg = df.groupby("speaker", as_index=False)["duration"].sum().sort_values("duration", ascending=False)
    fig = px.bar(agg, x="speaker", y="duration", title="Total Speaking Time (seconds)")
    return fig, agg


def utterance_length_boxplot(df: pd.DataFrame):
    """
    Box plot of utterance lengths (in words) per speaker.
    """
    df = df.copy()
    df["utterance_length"] = df["text"].fillna("").apply(lambda x: len(x.split()))
    fig = px.box(df, x="speaker", y="utterance_length",
                 title="Distribution of Utterance Lengths per Speaker")
    return fig, df


def pause_histogram(df: pd.DataFrame):
    """
    Histogram of pause durations (silence between utterances).
    """
    df_sorted = df.sort_values("start_time").reset_index(drop=True)
    pauses = df_sorted["start_time"].diff().dropna()
    pauses = pauses[pauses > 0]  # ignore overlaps
    fig = px.histogram(pauses, nbins=30, title="Distribution of Pause Durations (seconds)")
    return fig, pauses


def questions_per_speaker_bar(df: pd.DataFrame):
    """
    Bar chart of number of questions asked per speaker.
    Simple heuristic: count utterances ending with '?'.
    """
    df = df.copy()
    df["is_question"] = df["text"].fillna("").str.strip().str.endswith("?")
    agg = df.groupby("speaker")["is_question"].sum().reset_index(name="questions")
    fig = px.bar(agg, x="speaker", y="questions", title="Questions Asked per Speaker")
    return fig, agg


# def topic_distribution_chart(df: pd.DataFrame, topics_col: str = "topic"):
#     """
#     Topic distribution chart (treemap).
#     Requires a 'topic' column in df.
#     """
#     if topics_col not in df.columns:
#         raise ValueError("Dataframe must contain a 'topic' column for topic distribution.")
#     counts = df[topics_col].value_counts().reset_index()
#     counts.columns = ["topic", "count"]
#     fig = px.treemap(counts, path=["topic"], values="count", title="Topic Distribution")
#     return fig, counts


# def emotion_distribution_chart(df: pd.DataFrame, emotion_col: str = "emotion"):
#     """
#     Stacked bar chart of emotions per speaker.
#     Requires an 'emotion' column in df.
#     """
#     if emotion_col not in df.columns:
#         raise ValueError("Dataframe must contain an 'emotion' column for emotion distribution.")
#     agg = df.groupby(["speaker", emotion_col]).size().reset_index(name="count")
#     fig = px.bar(agg, x="speaker", y="count", color=emotion_col,
#                  title="Emotion Distribution per Speaker", barmode="stack")
#     return fig, agg


# def interruptions_heatmap(df: pd.DataFrame):
#     """
#     Heatmap of interruptions (overlapping speech).
#     Simplified heuristic: if one speaker's start_time < another's end_time and overlaps.
#     """
#     speakers = df["speaker"].unique()
#     inter_matrix = pd.DataFrame(0, index=speakers, columns=speakers)

#     for i, row1 in df.iterrows():
#         for j, row2 in df.iterrows():
#             if i >= j:  # avoid double counting
#                 continue
#             if row1["speaker"] != row2["speaker"]:
#                 overlap = max(0, min(row1["end_time"], row2["end_time"]) - max(row1["start_time"], row2["start_time"]))
#                 if overlap > 0:
#                     inter_matrix.loc[row1["speaker"], row2["speaker"]] += 1

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(inter_matrix, annot=True, cmap="Blues", fmt="d")
#     plt.title("Interruptions Heatmap (Who interrupts whom)")
#     fig = plt.gcf()
#     return fig, inter_matrix
# def interruptions_heatmap(df: pd.DataFrame):
#     """
#     Heatmap of interruptions (overlapping speech).
#     Simplified heuristic: if one speaker's start_time < another's end_time and overlaps.
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import pandas as pd

#     speakers = df["speaker"].unique()
#     inter_matrix = pd.DataFrame(0, index=speakers, columns=speakers)

#     for i, row1 in df.iterrows():
#         for j, row2 in df.iterrows():
#             if i >= j:
#                 continue
#             if row1["speaker"] != row2["speaker"]:
#                 overlap = max(0, min(row1["end_time"], row2["end_time"]) - max(row1["start_time"], row2["start_time"]))
#                 if overlap > 0:
#                     inter_matrix.loc[row1["speaker"], row2["speaker"]] += 1

    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(inter_matrix, annot=True, cmap="Blues", fmt="d", ax=ax)
    # ax.set_title("Interruptions Heatmap (Who interrupts whom)")
    # fig.tight_layout()

    # return fig, inter_matrix

