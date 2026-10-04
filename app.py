import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hybrid Recommendation Engine", layout="wide")

@st.cache_resource
def load_assets():
    ranker = joblib.load('models/lgbm_ranker.joblib')
    products = pd.read_csv('data/products.csv')
    
    # Precompute TF-IDF semantic similarities for Stage 1 Candidate Retrieval
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['title'] + " " + products['category'])
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return ranker, products, sim_matrix

ranker, df_products, sim_matrix = load_assets()

st.title("Two-Stage Hybrid Recommender & Re-Ranking Engine")
st.markdown("Combines **Semantic Content Matching (Stage 1)** with **LightGBM Click Propensity Re-Ranking (Stage 2)**.")

st.sidebar.header("User Context & Preferences")
user_session_duration = st.sidebar.slider("Session Duration (seconds)", 10, 600, 180)
user_price_sensitivity = st.sidebar.slider("Price Sensitivity (0 = Luxury, 1 = Budget)", 0.0, 1.0, 0.4)

selected_product = st.selectbox(
    "Select an item currently being viewed by user:",
    df_products['title'].values
)

col1, col2 = st.columns(2)

prod_idx = df_products[df_products['title'] == selected_product].index[0]
prod_row = df_products.iloc[prod_idx]

with col1:
    st.subheader("Currently Viewing")
    st.info(f"**{prod_row['title']}**\n\n* Category: {prod_row['category']}\n* Price: ${prod_row['price']}\n* Rating: {prod_row['rating']} ⭐")

# Stage 1: Candidate Generation (Top 25 nearest items)
sim_scores = list(enumerate(sim_matrix[prod_idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]
candidate_indices = [i[0] for i in sim_scores]
candidates = df_products.iloc[candidate_indices].copy()

# Stage 2: LightGBM Scoring & Re-Ranking
scoring_df = candidates.copy()
scoring_df['user_session_duration'] = user_session_duration
scoring_df['user_price_sensitivity'] = user_price_sensitivity

features = ['category', 'price', 'rating', 'popularity_score', 'user_session_duration', 'user_price_sensitivity']
click_probs = ranker.predict_proba(scoring_df[features])[:, 1]
candidates['predicted_ctr'] = click_probs

# Sort by predicted CTR
top_ranked = candidates.sort_values(by='predicted_ctr', ascending=False).head(5)

with col2:
    st.subheader("Top-5 Re-Ranked Recommendations")
    for _, item in top_ranked.iterrows():
        st.success(
            f"**{item['title']}** (Predicted CTR: {item['predicted_ctr']:.1%})\n\n"
            f"Category: {item['category']} | Price: ${item['price']} | Rating: {item['rating']} ⭐"
        )