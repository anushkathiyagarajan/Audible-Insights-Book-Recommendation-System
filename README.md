# 🎧 Audible Insights: Intelligent Book Recommendation System

## Project Overview

Audible Insights is a book recommendation system developed using machine learning techniques to provide personalized audiobook recommendations. The system analyzes audiobook information such as descriptions, genres, ratings, and review counts to recommend books similar to a user's interests.

The application also includes an interactive Exploratory Data Analysis (EDA) dashboard built using Streamlit.

---

## Objectives

- Analyze audiobook datasets and extract meaningful insights.
- Build recommendation systems using multiple approaches.
- Compare recommendation techniques.
- Develop an interactive Streamlit application for users.
- Visualize trends and insights through EDA.

---

##  Dataset Information

The project uses two audiobook datasets containing information such as:

- Book Name
- Author
- Rating
- Number of Reviews
- Price
- Description
- Listening Time
- Genre Information

The datasets were merged and cleaned before analysis.

---

## Methodology

### 1. Data Preparation
- Loaded and merged datasets.
- Removed duplicates.
- Handled missing values.
- Extracted genres.

### 2. Exploratory Data Analysis (EDA)
- Most Popular Genres
- Rating Distribution
- Top Rated Books
- Most Common Authors
- Ratings vs Review Counts
- Correlation Heatmap
- Outlier Analysis
- Publication Trend Limitation

### 3. Recommendation System Development

#### Content-Based Filtering
- TF-IDF Vectorization
- Cosine Similarity

#### Clustering-Based Recommendation
- K-Means Clustering

#### Hybrid Recommendation
- K-Means + Cosine Similarity

---

##  Model Comparison

| Method | Performance |
|----------|------------|
| Content-Based Filtering | Good |
| K-Means Clustering | Moderate |
| Hybrid Recommendation | Best |

The Hybrid Recommendation System was selected for deployment due to its superior recommendation quality.

---

##  Streamlit Application Features

###  Recommendations
- Select a favorite book.
- Receive Top 5 personalized recommendations.

### EDA Dashboard
- Most Popular Genres
- Rating Distribution
- Top Rated Books
- Most Common Authors
- Ratings vs Review Counts
- Correlation Heatmap
- Outlier Analysis

---

##  Project Structure

```
mini4/
│
├── app.py
├── notebooks/
│   ├── EDA.ipynb
│   └── ML.ipynb
│
├── models/
│   ├── final_books.csv
│   ├── cosine_similarity.pkl
│   ├── kmeans_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶ How to Run

### Clone Repository

```bash
git clone <repository-url>
cd mini4
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## Conclusion

The Audible Insights recommendation system successfully integrated EDA, NLP techniques, clustering algorithms, and recommendation models to provide meaningful audiobook recommendations. The Hybrid Recommendation approach produced the most relevant results and was deployed through an interactive Streamlit application.