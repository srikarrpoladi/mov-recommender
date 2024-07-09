import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Movie Searcher!!!", page_icon=":cinema:",layout="wide")
st.title(" :cinema: Movie Recommender")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>',unsafe_allow_html=True) ##number changes distance from that top and top padding is the function. 

df2 = pd.read_csv("cleaned_movie__final_df.csv")
df2["combined"] = df2["tagline"].fillna("") + " " + df2["description"].fillna("") + " " + df2["theme"].fillna("") + " " + df2["genre"].fillna("") + " " + df2["studio"].fillna("")

st.sidebar.header("Fillter your search")

language = st.sidebar.multiselect("Enter a language:", df2["language"].unique())
genre = st.sidebar.text_input("Enter a genre:")

# Filter the data based on user inputs
if language:
    df3 = df2[df2["language"].isin(language)]
else:
    df3 = df2.copy()

if genre:
    df4 = df3[df3["genre"].str.contains(genre, na=False)]
else:
    df4 = df3.copy()

# Handle the cases where both, one, or none of the filters are applied
if language and genre:
    filtered_df = df4
elif language and not genre:
    filtered_df = df3
elif not language and genre:
    filtered_df = df4
else:
    filtered_df = df2

# Display the filtered data
with st.expander("View Data"):
    st.write(filtered_df.iloc[:500].style.background_gradient(cmap="Oranges"))

title = st.text_input("Enter movie name:")
if title:
    list_of_all_titles = filtered_df["name"].tolist()
    find_close_match = difflib.get_close_matches(title, list_of_all_titles)
    if find_close_match:
        close_match = find_close_match[0]
        st.write("Did you mean:", close_match)
        
        # Proceed if a close match is found
        matched_row = filtered_df[filtered_df["name"] == close_match]["combined"].values[0]

        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(filtered_df["combined"])

        def search(matched_row):
            query_vec = vectorizer.transform([matched_row])
            similarity = cosine_similarity(query_vec, feature_vectors).flatten()
            indices = np.argpartition(similarity, -10)[-10:]
            valid_indices = indices[np.argsort(similarity[indices])][::-1]  # sort by similarity
            results = filtered_df.iloc[valid_indices]
            return results

        if st.button("Search"):
            st.write(search(matched_row))
    else:
        st.write("No close match found.")