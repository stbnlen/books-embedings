import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pinecone.init(api_key="bbf1ede2-3b65-43e2-af7a-0d7f2f9d234c", environment="gcp-starter")
index = pinecone.Index("books-embeddings")


def search(query, top_k):
    query_vector = model.encode(query).tolist()

    responses = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    # Format the responses for better display
    response_data = []
    for response in responses["matches"]:
        response_data.append(
            {
                "Title": response["metadata"]["Title"],
                "Description": response["metadata"]["description"],
                "Authors": response["metadata"]["authors"],
                "categories": response["metadata"]["categories"],
                "Year": response["metadata"]["publishedDate"],
                "Publisher": response["metadata"]["publisher"],
                "Score": response["score"],
            }
        )

    df = pd.DataFrame(response_data)
    return df


title = st.text_input("Movie title")

df = None

if title:
    df = search(title, 5)

if df is not None:
    st.dataframe(df)
