# Import Libraries
import streamlit as st
import openai
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from couchbase.search import SearchQuery, VectorQuery
import os

# Load env variables
CB_CONN_STR = st.secrets["COUCHBASE_CONN_STR"]
CB_USERNAME = st.secrets["COUCHBASE_USERNAME"]
CB_PASSWORD = st.secrets["COUCHBASE_PASSWORD"]
CB_BUCKET = st.secrets["COUCHBASE_BUCKET"]
CB_SCOPE = st.secrets["COUCHBASE_SCOPE"]
CB_COLLECTION = st.secrets["COUCHBASE_COLLECTION"]

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Connect to Couchbase

@st.cache_resource
def get_cb_cluster():
    auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
    options = ClusterOptions(
      authenticator=auth,
      kv_timeout=timedelta(seconds=20),
      config_idle_redial_timeout=timedelta(seconds=20)
    )
    cluster = Cluster(CB_CONN_STR, options)
    cluster.wait_until_ready(10)
    return cluster

cluster = get_cb_cluster()
bucket = cluster.bucket(CB_BUCKET)
scope = bucket.scope(CB_SCOPE)

# Generate Query Embedding

def generate_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def vector_search(query_embedding, top_k=5):
    vector_query = VectorQuery(
        field_name="embedding",
        vector=query_embedding,
        num_candidates=top_k
    )

    search_query = SearchQuery(
        index="idx_cvi_balanceposition",
        vector=vector_query
    )

    result = scope.search(search_query)

    docs = []
    for row in result.rows():
        docs.append(row.fields.get("content", ""))

    return docs

# RAG Prompt Construction

def build_prompt(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
You are an assistant answering based only on the context below.

Context:
{context_text}

Question:
{question}

Answer clearly and concisely.
"""
    return prompt

# LLM completion

def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# Streamlit UI layer

st.title("Couchbase Capella RAG Chatbot")

user_query = st.text_input("Ask a question")

if user_query:
    with st.spinner("Thinking..."):
        query_embedding = generate_embedding(user_query)
        retrieved_docs = vector_search(query_embedding)
        prompt = build_prompt(user_query, retrieved_docs)
        answer = generate_answer(prompt)

    st.subheader("Answer")
    st.write(answer)
