# 💡 Customer Feedback Analyzer — GPT Edition

A hybrid AI-powered tool that analyzes customer feedback data to uncover insights, sentiment trends, and actionable recommendations.  
Built to simulate a real-world **data product MVP**, this project demonstrates applied **data science**, **AI integration**, and **business intelligence** in one app.

---

## 🧠 Key Features
- **Automated Sentiment Analysis** — detects positive, negative, and neutral customer reviews.  
- **Executive Summaries** — generates concise, data-driven insights using local NLP and GPT hybrid logic.  
- **Keyword Extraction** — highlights recurring themes from customer feedback.  
- **Cluster Analysis** — groups similar reviews to identify root causes and emerging opportunities.  
- **Word Cloud Visualization** — displays dominant words for quick qualitative understanding.  
- **CSV Upload Support** — works with raw feedback datasets directly.  
- **Downloadable Reports** — provides summarized insights and visual exports.

---

## 🛠️ Tech Stack
| Layer | Tools / Frameworks |
|-------|--------------------|
| **Frontend** | Streamlit |
| **Backend / Core Logic** | Python 3.12 |
| **Data Science / NLP** | Scikit-learn, Transformers, NLTK |
| **Visualization** | Plotly, WordCloud |
| **Database** | SQLite (for local session tracking) |
| **Environment Management** | venv |

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Charles-git-N/customer-feedback-analyzer.git
cd customer-feedback-analyzer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py