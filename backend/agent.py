import streamlit as st
# from openai import OpenAI
import google.generativeai as genai
from backend.pipeline import enrich_transcript, compute_stats
from backend.visualization import (
    speaker_contribution_bar,
    speaker_contribution_pie,
    keyword_frequency_chart,
    keyword_mentions_table,
    sentiment_trend_line,
    conversation_flow_graph,
    # wordcloud_chart,
    keyword_timeline_chart,
    # sentiment_heatmap,
    speaker_turns_bar,
    conversation_timeline_chart,
    # wordcloud_per_speaker,
    action_items_table,
    speaker_network_graph,
    speaking_time_bar,
    utterance_length_boxplot,
    pause_histogram,
    questions_per_speaker_bar,
    # topic_distribution_chart,
    # emotion_distribution_chart,
    # interruptions_heatmap
)

import os
from dotenv import load_dotenv
import plotly.express as px
import re
import ast 
import pandas as pd


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  
genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --- Intent Mapping ---
INTENTS = {
    "speaker_contribution_bar": {
        "Description": (
            "A bar chart where each bar corresponds to a speaker and its height represents "
            "the total number of words spoken by that speaker in the meeting. "
            "This highlights differences in absolute speaking volume between participants "
            "and is best for comparing contributions side by side."
        ),
        "pipeline_fn": None,
        "viz_fn": speaker_contribution_bar
    },
    "speaker_contribution_pie": {
        "Description": (
            "A pie chart where each slice corresponds to a speaker, and slice size represents "
            "the proportion of total words spoken in the meeting. "
            "This highlights percentage-based contribution and overall balance of participation."
        ),
        "pipeline_fn": None,
        "viz_fn": speaker_contribution_pie
    },
    "keyword_frequency": {
        "Description": (
            "A bar chart showing the frequency of words or keywords across the transcript. "
            "The x-axis lists words, and the y-axis shows their occurrence counts. "
            "This highlights the most common or dominant terms and reveals recurring topics of discussion."
        ),
        "pipeline_fn": None,
        "viz_fn": keyword_frequency_chart
    },
    "keyword_mentions": {
        "Description": (
            "A table showing details of when and where a keyword was mentioned in the transcript. "
            "Each row includes: 'speaker' (who said it), 'start_time' (timestamp), and 'snippet' "
            "(short context around the keyword). This is useful for tracking the usage of specific terms "
            "and understanding their context in the conversation."
        ),
        "pipeline_fn": None,
        "viz_fn": keyword_mentions_table
    },
    "sentiment_trend": {
        "Description": (
            "A line chart plotting sentiment over time for each speaker. "
            "The x-axis represents meeting timeline, and the y-axis represents sentiment scores "
            "(such as positive, neutral, or negative values). "
            "This shows emotional dynamics, mood progression, and variations in tone throughout the discussion."
        ),
        "pipeline_fn": None,
        "viz_fn": sentiment_trend_line
    },
    "conversation_flow": {
        "Description": (
            "A directed graph showing conversation flow between speakers. "
            "Nodes represent speakers, and edges represent transitions where one speaker follows another. "
            "Edge weight or thickness can indicate frequency of transitions. "
            "This reveals interaction patterns, dominant speakers, and overall meeting structure."
        ),
        "pipeline_fn": None,
        "viz_fn": conversation_flow_graph
    },

    "speaking_time_bar": {
        "Description": (
            "A bar chart where each bar represents the total speaking time of a speaker in seconds. "
            "This highlights how much airtime each participant consumed in absolute terms, "
            "useful for identifying dominant voices."
        ),
        "pipeline_fn": None,
        "viz_fn": speaking_time_bar
    },
    "average_utterance_length": {
        "Description": (
            "A box plot showing the distribution of utterance lengths (in words) per speaker. "
            "This reveals which participants tend to speak in longer or shorter sentences, "
            "and highlights variation in speaking styles."
        ),
        "pipeline_fn": None,
        "viz_fn": utterance_length_boxplot
    },
    "pause_distribution": {
        "Description": (
            "A histogram showing the distribution of pauses (silence durations) between utterances. "
            "This reveals meeting pacing, responsiveness, and whether there were frequent interruptions."
        ),
        "pipeline_fn": None,
        "viz_fn": pause_histogram
    },
    "questions_per_speaker": {
        "Description": (
            "A bar chart showing the number of questions asked by each speaker. "
            "This highlights engagement style and identifies who leads the questioning "
            "or drives the conversation."
        ),
        "pipeline_fn": None,
        "viz_fn": questions_per_speaker_bar
    },
    # "topic_distribution": {
    #     "Description": (
    #         "A treemap or bar chart showing the distribution of clustered topics across the meeting. "
    #         "Each block represents a topic, sized by frequency of mentions. "
    #         "This highlights what subjects dominated the discussion."
    #     ),
    #     "pipeline_fn": None,
    #     "viz_fn": topic_distribution_chart
    # },
    # "emotion_distribution": {
    #     "Description": (
    #         "A stacked bar chart showing the distribution of different emotions "
    #         "(joy, sadness, anger, fear, neutral) per speaker. "
    #         "This highlights not just sentiment but the emotional tone of each participant."
    #     ),
    #     "pipeline_fn": None,
    #     "viz_fn": emotion_distribution_chart
    # },
    # "interruptions_heatmap": {
    #     "Description": (
    #         "A heatmap showing how often one speaker interrupts another, based on overlapping timestamps. "
    #         "Rows = interrupter, Columns = interrupted. "
    #         "This reveals dynamics of dominance, cooperation, or conflict."
    #     ),
    #     "pipeline_fn": None,
    #     "viz_fn": interruptions_heatmap
    # },

    "conversation_timeline": {
        "Description": (
            "A Gantt-style timeline chart showing when each speaker was talking over the course of the meeting. "
            "The x-axis is time, the y-axis lists speakers, and horizontal bars represent speaking turns. "
            "This clearly shows the rhythm of the meeting and who dominated which parts."
        ),
        "pipeline_fn": None,
        "viz_fn": conversation_timeline_chart
    },
    # "wordcloud_per_speaker": {
    #     "Description": (
    #         "A word cloud visualization for each speaker, showing their most frequently used words. "
    #         "Word size indicates frequency. This makes it easy to spot each speaker's focus or key topics."
    #     ),
    #     "pipeline_fn": None,
    #     "viz_fn": wordcloud_per_speaker
    # },
    "action_items_table": {
        "Description": (
            "A table listing action items or decisions extracted from the transcript. "
            "Each row includes: 'speaker', 'timestamp', and 'action item text'. "
            "This makes it easy to review follow-ups and responsibilities."
        ),
        "pipeline_fn": None,
        "viz_fn": action_items_table
    },
    # "sentiment_heatmap": {
    #     "Description": (
    #         "A heatmap showing sentiment intensity across time segments for each speaker. "
    #         "The x-axis is time bins, y-axis is speakers, and color represents average sentiment score. "
    #         "This reveals when mood shifts occurred and who contributed to them."
    #     ),
    #     "pipeline_fn": None,
    #     "viz_fn": sentiment_heatmap
    # },
    "speaker_network": {
        "Description": (
            "A network graph showing interaction intensity between speakers. "
            "Nodes represent speakers, and edges represent how often one directly follows another. "
            "Edge thickness represents frequency. This highlights pairwise dynamics and central figures."
        ),
        "pipeline_fn": None,
        "viz_fn": speaker_network_graph
    },
    # "wordcloud_chart": {
    #     "Description": (
    #         "A word cloud visualization of the most common words in the entire transcript. "
    #         "Each word's size corresponds to its frequency. "
    #         "This provides an immediate visual summary of dominant topics or recurring terms in the meeting."
    #     ),
    #     "pipeline_fn": None,
    #     "viz_fn": wordcloud_chart
    # },

    "keyword_timeline_chart": {
        "Description": (
            "A scatter/timeline chart showing when a specific keyword was mentioned by speakers throughout the meeting. "
            "The x-axis represents time, the y-axis represents speakers, and each point corresponds to an utterance containing the keyword. "
            "This helps track the discussion of specific terms and understand who mentioned them and when."
        ),
        "pipeline_fn": None,
        "viz_fn": keyword_timeline_chart
    },

    "speaker_turns_bar": {
        "Description": (
            "A bar chart where each bar represents the total number of speaking turns taken by a speaker. "
            "This highlights participation patterns and identifies speakers who dominated the conversation or spoke less frequently."
        ),
        "pipeline_fn": None,
        "viz_fn": speaker_turns_bar
    },


}


def classify_intent(query: str) -> str:
    
    intent_keys = list(INTENTS.keys())
    
    intent_descs = "\n".join(
        [f"- {k}: {v['Description']}" for k, v in INTENTS.items()]
    )
    
    prompt = f"""
    You are an expert intent-classification assistant for a meeting-analysis application.

    Your task: Read a single user query (the Query) and return exactly one matching intent label from the list below.

    Valid labels: {intent_keys} + ["generative"]

    ***** RULES *****
    1. Output must be a single plain-text label:
    - One of the exact keys from {intent_keys}
    - OR exactly "generative" if no label clearly fits the Query
    - No quotes, punctuation, explanations, or formatting.
    2. Never output multiple labels. Never invent labels. Never change spelling or casing.

    ***** WHEN TO CHOOSE "generative" *****
    - The query is vague, conversational, off-topic, or unrelated to any visualization.
    - The query asks for tasks, instructions, summaries, or anything beyond the scope of speaker/keyword/sentiment/conversation visualizations.
    - If the model is unsure which intent truly matches, prefer "generative" over guessing.

    ***** INTENT DESCRIPTIONS *****
    {intent_descs}

    ***** OUTPUT REMINDER *****
        Only return the label. No extra text.

    Query: {query}
    """

    response = gemini_model.generate_content(prompt)
    return response.text.strip()





# def run_agent(query: str, df):
#     intent = classify_intent(query).strip().lower().replace(" ", "_")

#     st.write(f"Detected intent: `{intent}`")

#     if intent in INTENTS:
#         viz_fn = INTENTS[intent]["viz_fn"]

#         data = df
        
#         fig_or_table, extra = viz_fn(data)

#         if intent == "keyword_mentions":
#             st.write(fig_or_table)
#         elif intent == "conversation_flow":
#             st.graphviz_chart(fig_or_table)
#         else:
#             st.plotly_chart(fig_or_table)
#     else:
#         st.info("⚡ Using generative path for this query...")

def run_agent(query: str, df):
    intent = classify_intent(query).strip().lower().replace(" ", "_")
    st.write(f"Detected intent: {intent}")

    if intent in INTENTS:
        viz_fn = INTENTS[intent]["viz_fn"]

        try:
            # Handle special cases where extra input is required
            if intent in ["keyword_mentions", "keyword_timeline_chart"]:
                # Ask user for keyword interactively
                keyword = st.text_input("Enter a keyword to analyze:")
                if not keyword:
                    st.warning("Please enter a keyword to continue.")
                    return
                result = viz_fn(df, keyword)

            elif intent in ["topic_distribution", "emotion_distribution"]:
                # These functions expect an extra column name
                colname = "topic" if intent == "topic_distribution" else "emotion"
                if colname not in df.columns:
                    st.error(f"Missing required column: {colname}")
                    return
                result = viz_fn(df, topics_col=colname) if intent == "topic_distribution" else viz_fn(df, emotion_col=colname)

            else:
                # Default case: only dataframe input
                result = viz_fn(df)

            # --- Normalize outputs ---
            if isinstance(result, tuple):
                output, extra = result
            else:
                output, extra = result, None

            # --- Render depending on type ---
            if isinstance(output, pd.DataFrame):
                st.dataframe(output)
            elif isinstance(output, str):  # Graphviz DOT string
                st.graphviz_chart(output)
            else:  # Plotly or Matplotlib fig
                try:
                    st.plotly_chart(output, use_container_width=True)
                except Exception:
                    st.pyplot(output)

        except Exception as e:
            st.error(f"Error running visualization '{intent}': {e}")

    else:
        st.info("⚡ Using generative path for this query...")
        run_generative_path(query, df)


def extract_code(text):
    match = re.search(r"```(?:python)?\n([\s\S]+?)```", text, re.IGNORECASE)
    return match.group(1) if match else text

def triple_quote_balanced(code):
    return code.count('"""') % 2 == 0 and code.count("'''") % 2 == 0

def sanitize_code(code):
    # Remove all triple-quoted strings: both """ and '''
    code = re.sub(r'"""[\s\S]*?"""', '""', code)
    code = re.sub(r"'''[\s\S]*?'''", "''", code)
    return code

def run_generative_path(query: str, df):
    """Ask LLM to write custom Streamlit+Plotly code."""
    prompt = f"""
You are an expert Python coding assistant.
Your role is to generate error-free, production-ready visualization code for meeting transcripts.

-------------------------------------------------
CONTEXT
- The transcript is preloaded as a Pandas DataFrame named df.
- The DataFrame has these exact columns: {list(df.columns)}.
- The user provides a natural language query (the Query).
- The output must be Streamlit + Plotly Express code that directly answers the query.

-------------------------------------------------
ABSOLUTE RULES (must always follow)
1. Output must be complete Python code only.
   - Do NOT include explanations, comments, Markdown, or text outside Python code.
   - The output must be a self-contained script that runs directly with streamlit run.

2. String usage rules:
   - Use only single-line strings with double quotes "...".
   - NEVER use triple quotes (''' or \""") anywhere in the code.
   - Do NOT leave strings open or incomplete.

3. Imports (always required at the top):
   import streamlit as st
   import plotly.express as px
   import pandas as pd

4. Figure rendering:
   - Every visualization must end with:
     st.plotly_chart(fig, use_container_width=True)
   - If a table is required, use st.dataframe(...).

5. Column validation:
   - Before using a column in df, check if it exists in {list(df.columns)}.
   - If the user requests a non-existent column, instead output:
     st.write("Error: Requested column not found in DataFrame")

6. Chart selection (must match query intent):
   - Counts, comparisons → px.bar
   - Proportions → px.pie
   - Trends / over time → px.line
   - Sentiment change / numerical progression → px.line
   - Speaker contribution → px.bar or px.pie depending on user request
   - Keyword frequency → px.bar
   - Conversation flow or structure (if not supported by px) → fallback: st.dataframe(df)

7. Fallback behavior:
   - If the Query is vague (e.g., "show me the data"), return:
     st.dataframe(df)

8. Completeness check:
   - The final code must always:
     * Import required libraries.
     * Reference only existing df columns.
     * Build the figure using Plotly Express.
     * Display it using Streamlit.
   - No placeholders, no TODOs, no incomplete code.

-------------------------------------------------
OUTPUT FORMAT
- Return only the complete Python script, nothing else.
- No comments, no explanations, no Markdown formatting.

-------------------------------------------------
USER QUERY
{query}
"""


    response = gemini_model.generate_content(prompt)
    code_raw = response.text.strip()
    code = extract_code(code_raw)

    if not triple_quote_balanced(code):
        st.error("Generated code contains unbalanced triple-quoted strings.")
    else:
        code = sanitize_code(code)
        try:
            ast.parse(code)
            exec(code, {"df": df, "st": st, "px": px})
        except SyntaxError as e:
            st.error(f"Syntax error in generated code: {e}")
        except Exception as e:
            st.error(f"Error running generated code: {e}")
