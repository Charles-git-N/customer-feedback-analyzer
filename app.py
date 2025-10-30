# 💡 Customer Feedback Analyzer — GPT Hybrid Edition (Full MVP Version)
import os
import json
import time
import re
import sqlite3
import tempfile
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
from fpdf import FPDF

# ------------------------------------------
# 🧠 Setup + Config
# ------------------------------------------
st.set_page_config(page_title="💡 Customer Feedback Analyzer — GPT Edition", page_icon="💬", layout="wide")

st.markdown("""
<style>
body, .stApp { background-color: #0b1220; color: #e6eef8; }
h1, h2, h3, h4, h5 { color: #ffffff; }
.stButton>button { background-color: #2563eb; color: white; border-radius: 6px; border: none; }
.stButton>button:hover { background-color: #1e40af; }
</style>
""", unsafe_allow_html=True)

st.title("💡 Customer Feedback Analyzer — GPT Edition (Hybrid MVP)")
st.write("Analyze reviews, generate insights, and create executive summaries — with local NLP and GPT combined.")

# ------------------------------------------
# 🔑 Authentication (Simple)
# ------------------------------------------
user_name = st.sidebar.text_input("Enter your name (for personalized reports):", value="")
if not user_name:
    st.warning("👋 Please enter your name in the sidebar to begin.")
    st.stop()

# ------------------------------------------
# ⚙️ Settings + Sidebar
# ------------------------------------------
uploaded = st.sidebar.file_uploader("Upload your CSV or TXT file", type=["csv", "txt"])
n_clusters = st.sidebar.slider("Number of topic clusters", 2, 8, 4)
use_gpt = st.sidebar.checkbox("Use GPT for summary & recommendations", value=True)
show_wordcloud = st.sidebar.checkbox("Show word cloud", True)
save_history = st.sidebar.checkbox("Save analysis history", True)
gpt_model = st.sidebar.selectbox("GPT Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY) if use_gpt and OPENAI_KEY else None
if use_gpt and not OPENAI_KEY:
    st.sidebar.error("⚠️ OPENAI_API_KEY not set. GPT will be disabled.")
    use_gpt = False

# ------------------------------------------
# 🧩 Functions
# ------------------------------------------
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    s = analyzer.polarity_scores(text)
    comp = s["compound"]
    if comp >= 0.5:
        return "POSITIVE", comp
    elif comp <= -0.5:
        return "NEGATIVE", comp
    else:
        return "NEUTRAL", comp

def cluster_reviews(texts, k=4):
    vect = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vect.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    return labels, vect, km

def extract_keywords_by_cluster(texts, labels, top_n=5):
    cluster_keywords = {}
    for cluster in sorted(set(labels)):
        cluster_text = " ".join([t for t, l in zip(texts, labels) if l == cluster])
        words = re.findall(r'\b[a-z]{3,}\b', cluster_text.lower())
        common = Counter(words).most_common(top_n)
        cluster_keywords[cluster] = [w for w, _ in common]
    return cluster_keywords

def gpt_summary_prompt(data):
    prompt = f"""
You are a business analyst. Given the following summarized customer reviews data:
{json.dumps(data[:50])}

Return a JSON with:
- overall_sentiment
- top_themes (3)
- recommended_actions (3)
- one_paragraph_summary (3-4 sentences, professional tone)
"""
    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a concise and professional data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        text = response.choices[0].message.content
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end == -1:
            return {"error": "Invalid GPT response", "raw": text}
        return json.loads(text[start:end])
    except Exception as e:
        return {"error": str(e)}

def save_to_db(user, df):
    conn = sqlite3.connect("analysis_history.db")
    df["user"] = user
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_sql("feedback_data", conn, if_exists="append", index=False)
    conn.close()

def generate_pdf(user, summary_data, sentiment_chart_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"Customer Feedback Report — {user}", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 8, summary_data.get("one_paragraph_summary", "No summary available"))
    pdf.ln(10)
    pdf.image(sentiment_chart_path, x=10, w=180)
    path = os.path.join(tempfile.gettempdir(), "feedback_report.pdf")
    pdf.output(path)
    return path

# ------------------------------------------
# 🚀 MAIN
# ------------------------------------------
if uploaded:
    st.success(f"✅ File uploaded by {user_name}")

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame({"review": uploaded.read().decode().splitlines()})

    df.columns = [c.lower() for c in df.columns]
    if "review" not in df.columns:
        st.error("Your file must include a 'review' column.")
        st.stop()

    df = df.dropna(subset=["review"]).reset_index(drop=True)
    st.info(f"Loaded {len(df)} reviews")

    # Local Sentiment Analysis
    with st.spinner("Analyzing sentiments..."):
        sentiments, scores = zip(*[analyze_sentiment(r) for r in df["review"]])
        df["sentiment"], df["compound"] = sentiments, scores

    # Clustering
    with st.spinner("Clustering reviews into topics..."):
        labels, vect, model = cluster_reviews(df["review"].tolist(), k=n_clusters)
        df["cluster"] = labels
        cluster_keywords = extract_keywords_by_cluster(df["review"], labels)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Insights", "🧠 Executive Report", "📄 Export"])

    # TAB 1 - Dashboard
    with tab1:
        counts = df["sentiment"].value_counts()
        st.metric("Total Reviews", len(df))
        col1, col2 = st.columns(2)
        col1.bar_chart(counts)
        col2.dataframe(df.head(10))

        if show_wordcloud:
            st.subheader("Word Cloud")
            all_text = " ".join(df["review"])
            wc = WordCloud(width=800, height=300, background_color="white").generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # TAB 2 - Insights
    with tab2:
        st.subheader("📂 Cluster Topics & Key Insights")

        for c in sorted(set(labels)):
            st.markdown(f"### 🧩 Cluster {c+1}")
            st.markdown(f"**Top Keywords:** {' | '.join(cluster_keywords[c])}")

            examples = df[df["cluster"] == c]["review"].head(3).tolist()
            st.markdown("**Example Reviews:**")
            for e in examples:
                st.markdown(f"- {e}")

            sample_text = " ".join(examples)
            local_sentiment, _ = analyze_sentiment(sample_text)
            st.markdown(f"**Cluster Sentiment:** {local_sentiment}")

            if use_gpt:
                with st.spinner(f"Summarizing Cluster {c+1}..."):
                    cluster_result = gpt_summary_prompt(examples)
                if "error" not in cluster_result:
                    st.markdown(f"**Summary:** {cluster_result.get('one_paragraph_summary', 'N/A')}")
                else:
                    st.warning("⚠️ GPT failed to summarize this cluster.")
            
            st.markdown("---")
    # TAB 3 - Executive Report (GPT)
    with tab3:
        if use_gpt:
            with st.spinner("Generating executive summary..."):
                result = gpt_summary_prompt(df["review"].tolist())
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(f"### Overall Sentiment: {result.get('overall_sentiment', 'N/A')}")
                st.markdown("#### Key Themes")
                for theme in result.get("top_themes", []):
                    st.write(f"- {theme}")
                st.markdown("#### Recommended Actions")
                for action in result.get("recommended_actions", []):
                    st.write(f"- {action}")
                st.markdown("#### Executive Summary")
                st.info(result.get("one_paragraph_summary", ""))
        else:
            st.warning("Enable GPT in sidebar for executive summaries.")

    # TAB 4 - Export
    with tab4:
        if save_history:
            save_to_db(user_name, df)
            st.success("Analysis saved to history database (SQLite).")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Processed CSV", data=csv_data, file_name="feedback_analysis.csv", mime="text/csv")
        st.write("Generate PDF summary (includes executive report and sentiment chart)")
        # save chart
        fig = px.histogram(df, x="sentiment", color="sentiment", title="Sentiment Distribution")
        chart_path = os.path.join(tempfile.gettempdir(), "sentiment_chart.png")
        try:
            fig.write_image(chart_path)
        except Exception as e:
            st.warning(f"Chart export skipped — {e}")
            chart_path = None
        if use_gpt:
            pdf_path = generate_pdf(user_name, result, chart_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download PDF Report", data=f, file_name="executive_report.pdf")

else:
    st.info("👆 Upload a CSV or TXT file to begin your analysis.")