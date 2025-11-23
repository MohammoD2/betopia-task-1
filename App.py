# streamlit_icp_app.py
import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch
import json
from typing import List, Dict

# --- Streamlit App ---
st.set_page_config(page_title="ICP Matching Engine", layout="wide")
st.title("🔥 Advanced ICP Matching Engine")

DATA_PATH = "Data/Task-1-data(130 company).csv"   # Fixed data path


# --- MMR Explanation Function ---
def generate_mmr_explanation(
    query_description: str,
    company_description: str,
    model: SentenceTransformer,
    top_n: int = 4
) -> str:

    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s',
        company_description
    )
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return company_description.strip()

    sentence_embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    query_embedding = model.encode(
        query_description,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    relevance_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    selected_indices = []
    best_idx = torch.argmax(relevance_scores).item()
    selected_indices.append(best_idx)

    for _ in range(top_n - 1):
        candidate_indices = [
            i for i in range(len(sentences))
            if i not in selected_indices
        ]
        if not candidate_indices:
            break

        max_mmr = -float("inf")
        best_candidate_idx = -1
        selected_embeds = sentence_embeddings[selected_indices]

        for idx in candidate_indices:
            candidate_embed = sentence_embeddings[idx].unsqueeze(0)
            redundancy = util.cos_sim(candidate_embed, selected_embeds).max().item()
            mmr_score = 0.6 * relevance_scores[idx].item() - 0.4 * redundancy

            if mmr_score > max_mmr:
                max_mmr = mmr_score
                best_candidate_idx = idx

        if best_candidate_idx != -1:
            selected_indices.append(best_candidate_idx)

    selected_sentences = [sentences[i] for i in sorted(selected_indices)]
    explanation = " ".join(selected_sentences)
    return f"This company matches the ICP because: {explanation}"


# --- Sidebar Configuration ---
st.sidebar.header("Configuration (Step 1)")

st.sidebar.write("**Data Path:** `Data/Task-1-data(130 company).csv` (Auto Loaded)")

model_option = st.sidebar.selectbox("Select Embedding Model", [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5"
])

icp_description = st.sidebar.text_area(
    "ICP Description",
    value="We are looking for e-commerce and fashion companies in Bangladesh with scalable online presence.",
    height=120
)

top_n = st.sidebar.number_input(
    "Number of sentences for explanation",
    min_value=1,
    max_value=10,
    value=4
)

process_button = st.sidebar.button("🚀 PROCESS")


# --- MAIN PROCESSING ---
if process_button:

    st.success("Processing started...")

    # Load data
    df = pd.read_csv(DATA_PATH)
    st.info(f"Loaded {len(df)} companies.")

    # Combine descriptions directly
    df['combined_features'] = df.apply(
        lambda r: f"{str(r['short_description'])} Industry: {str(r['industry'])}. Tech: {str(r.get('tech_stack', ''))}.",
        axis=1
    )

    # Load model
    with st.spinner(f"Loading embedding model: {model_option} ..."):
        model = SentenceTransformer(model_option)

    # Encode embeddings
    with st.spinner("Encoding company descriptions..."):
        company_features = df['combined_features'].tolist()
        company_embeddings = model.encode(
            company_features,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        icp_embedding = model.encode(icp_description, convert_to_tensor=True)

    # Similarity
    cosine_scores = util.cos_sim(icp_embedding, company_embeddings)[0]
    df['similarity_score'] = cosine_scores.cpu().numpy()

    # Rank companies
    df_ranked = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)

    # Generate explanations
    with st.spinner("Generating explanations for top companies..."):
        results: List[Dict] = []

        for _, row in df_ranked.iterrows():
            explanation = generate_mmr_explanation(
                icp_description,
                str(row['short_description']),
                model,
                top_n=top_n
            )
            results.append({
                "company_name": row['company_name'],
                "similarity_score": round(row['similarity_score'], 4),
                "explanation": explanation
            })

    # Show top 10
    st.subheader("Top 10 Matching Companies")
    top10_df = pd.DataFrame(results[:10])
    st.dataframe(top10_df)

    # JSON download
    json_output = json.dumps(results, indent=4)
    st.download_button(
        "📥 Download Full JSON Output",
        data=json_output,
        file_name="icp_ranked_companies.json",
        mime="application/json"
    )


else:
    st.warning("Click the **Process** button after configuring the inputs.")
