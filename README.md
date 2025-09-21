# VORTEX

*Transform Conversations into Actionable Insights Instantly*

![Last Commit](https://img.shields.io/badge/last%20commit-today-brightgreen)
![Python](https://img.shields.io/badge/python-100%25-blue)
![Languages](https://img.shields.io/badge/languages-1-lightgrey)

## Built with the tools and technologies:

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7B93E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-0A5DAB?style=for-the-badge&logo=spacy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-7C4DFF?style=for-the-badge&logo=google&logoColor=white)

Vortex is a *Meeting Transcript Analysis * powered by *Streamlit, **NLP pipelines, and a **hybrid AI agent*.
It allows you to upload a meeting transcript (JSON format) and explore insights such as:

* *Speaker contributions* (who talked the most, speaking turns, time distribution).
* *Keyword analysis* (frequency, mentions, timelines, word clouds).
* *Sentiment* (line charts, heatmaps, distributions).
* *Conversation dynamics* (flow graphs, interruptions, networks).

The system is designed as a *hybrid LLM + deterministic pipeline model*:

* ✅ *Deterministic Path* → For common meeting insights, mapped to pre-coded visualizations.
* ⚡ *Generative Path* → For custom or unusual queries, LLM generates code dynamically.

This ensures *reliability* for standard tasks but also *flexibility* for exploratory analysis.

---

## 🚀 Features

* Upload meeting transcript JSON (local MVP, no cloud storage).
* Preprocessing pipeline for:

  * Parsing transcript into DataFrame.
  * Extracting keywords, relations, NER.
  * Sentiment analysis (utterance-level + trends).
  * Conversation stats.
* Pre-built visualizations:

  * Bar, pie, line, heatmap, network, word cloud, Gantt-style timeline.
* Interactive *LLM Agent*:

  * Maps queries to known intents (speaker_contribution_bar, keyword_frequency, etc.).
  * Falls back to *generative visualization code* for new queries.
* Extendable — add new pipeline functions, visualizations, or intents easily.

---
| Feature               | Type of Visualization  | Demo Image                                                                 |
|-----------------------|----------------------|---------------------------------------------------------------------------|
| Speaker Contribution  | Bar Chart             | ![Speaker Contribution](https://raw.githubusercontent.com/mohit-79/Vortex/master/Demo/Screenshot%202025-09-21%20132119.png) |
| Sentiment Trend       | Line Chart            | ![Sentiment Trend](https://raw.githubusercontent.com/mohit-79/Vortex/master/Demo/Screenshot%202025-09-21%20132356.png) |
| Conversation Flow     | Network               | ![Conversation Flow](https://raw.githubusercontent.com/mohit-79/Vortex/master/Demo/Screenshot%202025-09-21%20132545.png) |
| Interaction Network   | NetworkX Graph        | ![Interaction Network](https://raw.githubusercontent.com/mohit-79/Vortex/master/Demo/Screenshot%202025-09-21%20132656.png) |


---

## 📂 Project Structure


transcript-ai-agent/
│
├── backend/
│   ├── pipeline.py          # Data processing & enrichment
│   ├── visualization.py     # Pre-coded visualizations
│   └── agent.py             # Intent classification & hybrid agent
│
├── app/
|    |- main.py              # Streamlit frontend
├── requirements.txt         # Python dependencies
├── .env                     # API key (Gemini/OpenRouter)
└── README.md                # Documentation


---

## ⚙ Setup Instructions

### 1. Clone the Project

bash
git clone https://github.com/yourusername/transcript-ai-agent.git
cd transcript-ai-agent


### 2. Create Virtual Environment

bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows


### 3. Install Requirements

bash
pip install -r requirements.txt


### 4. Set Environment Variables

Create a .env file:

env
# If using Gemini
GEMINI_API_KEY=your_gemini_key_here

# OR if using OpenRouter
OPENROUTER_API_KEY=your_openrouter_key_here


### 5. Run Streamlit App

bash
streamlit run app.py


---

## 📊 Transcript Format

The transcript must be a JSON file with utterances. Example:

json
[
  {
    "speaker": "Alice",
    "start_time": 0,
    "end_time": 5,
    "text": "Welcome everyone to the project kickoff meeting."
  },
]

---

## 🏗 Architecture

### 🔹 Pipeline (pipeline.py)

* **parse_transcript** → Load JSON → DataFrame.
* **enrich_transcript** → Add keywords, sentiment, NER, relations.
* **compute_sentiment_trends** → Speaker-level mood shifts.
* **compute_stats** → Aggregate stats (word counts, durations, etc.).

👉 This ensures every transcript is *normalized* before visualization.

---

### 🔹 Agent (agent.py)

1. *Classify intent*:

   * Maps query → predefined INTENTS.
   * If no match → "generative".
2. *Run deterministic path*:

   * Executes visualization function directly.
3. *Run generative path*:

   * Uses LLM to *generate new Streamlit + Plotly code* dynamically.
   * Code is validated and executed in sandbox.

👉 Hybrid design = *safe defaults + flexibility*.

---

## 📊 Available Visualizations

### Speaker Analysis

* Contribution (bar/pie).
* Speaking time (seconds).
* Number of speaking turns.
* Utterance length distribution.
* Questions asked per speaker.

### Keywords

* Keyword frequency.
* Keyword timeline.

### Sentiment & Emotions

* Sentiment trend (line).

### Conversation Dynamics

* Conversation flow (Graphviz).
* Conversation timeline (Gantt).
* Speaker interaction network.
---



## ⚡ Tech Stack

* *Frontend*: Streamlit
* *Visualization*: Plotly, Matplotlib, Seaborn, WordCloud, Graphviz, NetworkX
* *NLP*: spaCy (relations, NER), sentiment analysis
* *LLM API*: Gemini / OpenRouter
* *Backend*: Python (pandas, regex, heuristics)

---

## 🛡 Hybrid Model Explanation

### Deterministic Path

* Query → maps to fixed function.
* Always produces *stable, correct charts*.
* Example: "Who talked the most?" → speaker_contribution_bar.

### Generative Path

* If query doesn’t match → ask LLM to write code.
* Sandbox executes *custom visualization code*.
* Example: "Show me a bubble chart of sentiment vs word count."

👉 This ensures MVP is *both robust and flexible*.

---



## 👨‍💻 Contributors
* *Mohit Gunani     Mohit Harjani     Soham Labhshetwar* 🎯 (Project creator)
---

## 📜 License

MIT License. Free to use and modify.
