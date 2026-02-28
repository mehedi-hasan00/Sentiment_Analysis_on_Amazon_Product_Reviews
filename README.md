# 📦 Amazon Product Review Sentiment Analysis

A Machine Learning based **Natural Language Processing (NLP)** project designed to analyze and classify customer feedback. This model utilizes **Bernoulli Naive Bayes** to identify sentiments with exceptional precision, helping businesses understand customer satisfaction.

🌐 **Live App:** [https://sentiment-analysis-on-amazon.streamlit.app/](https://sentiment-analysis-on-amazon.streamlit.app/)

## 🚀 Live Demo
Run the Streamlit app to input any product review and get instant sentiment classification.
* **Input:** "The product quality is amazing and it arrived on time!"
* **AI Output:** This is a **Positive Review** ✅
* **Input:** "Worst experience, the item was broken and delivery was late."
* **AI Output:** This is a **Negative Review** ❌

## 🧠 Project Overview
The goal of this project is to build a robust sentiment classifier capable of distinguishing between positive and negative reviews from a large-scale Amazon dataset. 

By utilizing **NLP Pre-processing** and **Bernoulli Naive Bayes**, the model focuses on:
* **High Precision** – Reducing False Positives to maintain brand integrity.
* **Contextual Understanding** – Using N-grams to capture phrases like "not good" or "very happy".
* **Business Insight** – Accurately identifying negative feedback for crisis management.

## 📊 Model Performance & Comparison

We evaluated several models, and while others showed high recall, **Bernoulli NB** was selected for its superior **Precision**.

| Model | Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Bernoulli NB** | **0.899** | **0.960** | 0.904 |
| SVC | 0.908 | 0.917 | 0.965 |
| Logistic Regression | 0.903 | 0.909 | 0.968 |
| Random Forest | 0.883 | 0.887 | 0.970 |
| GBDT | 0.858 | 0.858 | 0.974 |

### 🏆 Why Bernoulli NB?
We prioritized **Precision (96.0%)** because, from a business perspective, misclassifying a negative complaint as "Positive" is more damaging than missing a few positive ones. Bernoulli NB effectively minimized False Positives (only 132 cases).

## 🛠️ Tech Stack & Architecture
Built using **Python** and **Scikit-Learn**:
* **Vectorization:** TF-IDF Vectorizer
* **N-grams:** (1, 2) - Unigrams & Bigrams
* **Max Features:** 5,000 most frequent terms
* **Pre-processing:** Lowercasing, Tokenization, Stopword removal, and Stemming (PorterStemmer).

## 📦 Installation & Setup

1️⃣ **Clone the Repository**
```bash
git clone [https://github.com/mehedi-hasan00/Sentiment_Analysis_on_Amazon_Product_Reviews.git](https://github.com/mehedi-hasan00/Sentiment_Analysis_on_Amazon_Product_Reviews.git)
cd Amazon-Sentiment-Analysis
```

2️⃣ **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Run the App**
```bash
streamlit run app.py
```

## 📂 Project Structure
```
├── main.py                      # Streamlit web app
├── model_training.ipynb         # Data analysis & model training
├── model.pkl                    # Exported Bernoulli NB Pipeline
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

## 📝 requirements.txt
```text
streamlit
pandas
numpy
scikit-learn
nltk
```

## 📈 Deep Learning Extension (LSTM)
We also experimented with **LSTM (Long Short-Term Memory)** to capture sequential patterns:
* **Accuracy:** 85.32%
* **Precision:** 92.40%
* **F1 Score:** 90.11%
* **Confusion Matrix:** `[[738, 220], [367, 2675]]`

## 👤 Author
**Mehedi Hasan**
🔗 **LinkedIn:** [https://www.linkedin.com/in/mehedi-hasan-094855388/](https://www.linkedin.com/in/mehedi-hasan-094855388/)
🔗 **Kaggle:** [https://www.kaggle.com/mehedi71](https://www.kaggle.com/mehedi71)