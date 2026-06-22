import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Audible Insights",
    page_icon="🎧",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

/* App Background */
.stApp {
    background: linear-gradient(to right, #F4F8FF, #EAF3FF);
}

/* Title */
h1 {
    color: #1E3C72;
    text-align: center;
    font-size: 3rem;
}

/* Subtitle */
h4 {
    color: #4A5568;
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

/* Dropdown */
div[data-baseweb="select"] {
    background-color: white;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background-color: #FF9900;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 220px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #E68A00;
    color: white;
}

/* Recommendation Cards */
.recommend-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    border-left: 6px solid #FF9900;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA AND MODELS
# =====================================

df = pd.read_csv("models/final_books.csv")

cosine_sim = joblib.load(
    "models/cosine_similarity.pkl"
)

# =====================================
# BOOK INDICES
# =====================================

indices = pd.Series(
    df.index,
    index=df["Book Name"]
)

# =====================================
# HYBRID RECOMMENDATION FUNCTION
# =====================================

def hybrid_recommendations(title, n=5):

    idx = indices[title]

    cluster = df.iloc[idx]["Cluster"]

    cluster_books = df[
        df["Cluster"] == cluster
    ].index.tolist()

    sim_scores = []

    for book_idx in cluster_books:

        if book_idx != idx:

            sim_scores.append(
                (
                    book_idx,
                    cosine_sim[idx][book_idx]
                )
            )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_books = [
        i[0]
        for i in sim_scores[:n]
    ]

    return df.loc[
        top_books,
        ["Book Name", "Author", "Genre"]
    ]

# =====================================
# SIDEBAR NAVIGATION
# =====================================

menu = st.sidebar.radio(
    "Navigation",
    ["🎧 Recommendations", "📊 EDA Dashboard"]
)

# =====================================
# RECOMMENDATIONS PAGE
# =====================================

if menu == "🎧 Recommendations":

    st.markdown(
        "<h1>🎧 Audible Insights</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4>Discover Your Next Favorite Audiobook</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    selected_book = st.selectbox(
        "📚 Choose a Book",
        sorted(df["Book Name"].unique())
    )

    st.write("")

    if st.button("✨ Recommend Books"):

        recommendations = hybrid_recommendations(
            selected_book
        )

        st.markdown(
            "<h2 style='color:#1E3C72;'>📚 Recommended Books</h2>",
            unsafe_allow_html=True
        )

        for _, row in recommendations.iterrows():
            
            st.markdown(f"""
        <div class="recommend-card">
            <h3>📖 {row['Book Name']}</h3>
            <p><strong>✍ Author:</strong> {row['Author']}</p>
            <p><strong>🏷 Genre:</strong> {row['Genre']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================
# EDA DASHBOARD
# =====================================

elif menu == "📊 EDA Dashboard":

    import matplotlib.pyplot as plt
    import seaborn as sns

    st.markdown(
        "<h1>📊 EDA Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4>Explore Insights from the Audible Dataset</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    # =====================================
    # MOST POPULAR GENRES
    # =====================================

    st.subheader("📚 Most Popular Genres")

    top_genres = (
        df["Genre"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_genres)

    st.write("")

    # =====================================
    # RATING DISTRIBUTION
    # =====================================

    st.subheader("⭐ Rating Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["Rating"],
        bins=10,
        edgecolor="black"
    )

    ax.set_title("Distribution of Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Books")

    st.pyplot(fig)

    st.write("")

    # =====================================
    # TOP RATED BOOKS
    # =====================================

    st.subheader("🏆 Top Rated Books")

    top_books = (
        df.sort_values(
            by="Rating",
            ascending=False
        )[
            ["Book Name", "Author", "Rating"]
        ]
        .head(10)
    )

    st.dataframe(
        top_books,
        use_container_width=True
    )

    st.write("")

    # =====================================
    # RATINGS VS REVIEW COUNTS
    # =====================================

    st.subheader("📈 Ratings vs Review Counts")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Number of Reviews"],
        df["Rating"],
        alpha=0.5
    )

    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Rating")
    ax.set_title("Relationship Between Ratings and Reviews")

    st.pyplot(fig)

    st.write(
        """
        **Insight:** Books with higher review counts generally maintain
        ratings between 4 and 5 stars. However, more reviews do not
        always guarantee higher ratings.
        """
    )

    st.write("")

    # =====================================
    # CORRELATION HEATMAP
    # =====================================

    st.subheader("🔥 Correlation Heatmap")

    corr = df[
        ["Rating", "Number of Reviews", "Price"]
    ].corr()

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Correlation Between Numerical Features")

    st.pyplot(fig)

    st.write("")

    # =====================================
    # PUBLICATION YEAR TRENDS
    # =====================================

    st.subheader("📅 Publication Trends")

    st.info(
        "Publication year information is not available in the provided dataset. Therefore, publication trends could not be analyzed."
    )

# =====================================
# FOOTER
# =====================================

st.write("")
st.write("")

st.markdown(
    """
    <hr>

    <center style='color:#666666;'>
    Built using ❤️ with Streamlit |
    Audible Book Recommendation System
    </center>
    """,
    unsafe_allow_html=True
)