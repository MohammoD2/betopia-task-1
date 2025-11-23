# --- Imports ---
import os
import re
import pandas as pd
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import json

# --- Configuration ---
COMPANY_DATA_PATH = "Data/Task-1-data(130 company).csv"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'


# --- Utility Functions ---
def generate_mmr_explanation(
    query_description: str,
    company_description: str,
    model: SentenceTransformer,
    top_n: int = 4
) -> str:
    """
    Generates a solid explanation using MMR sentence selection.
    """
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
    explanation = " ".join(selected_sentences).strip()

    return f"This company matches the ICP because: {explanation}"


# --- Main ICP Engine ---
def run_icp_engine():
    """Runs the advanced ICP similarity engine on CSV data."""

    if not os.path.exists(COMPANY_DATA_PATH):
        print(f"ERROR: File '{COMPANY_DATA_PATH}' not found.")
        return

    df = pd.read_csv(COMPANY_DATA_PATH)
    print(f"Loaded {len(df)} companies.")

    icp_description = (
        "We are looking for e-commerce and fashion companies in Bangladesh "
        "with scalable online presence."
    )
    print(f"\nICP Description:\n{icp_description}\n")

    # Directly use short_description, no cleaning
    df['combined_features'] = df.apply(
        lambda r: (
            f"{r['short_description']} "
            f"Industry: {r['industry']}. "
            f"Tech: {r.get('tech_stack', '')}. "
            f"Location: Bangladesh-focused if applicable."
        ),
        axis=1
    )

    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    company_features = df['combined_features'].tolist()
    print("Encoding company descriptions...")
    company_embeddings = model.encode(
        company_features,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32
    )
    icp_embedding = model.encode(icp_description, convert_to_tensor=True)

    cosine_scores = util.cos_sim(icp_embedding, company_embeddings)[0]
    df['similarity_score'] = cosine_scores.cpu().numpy()

    df_ranked = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)

    results: List[Dict] = []

    print("Generating explanations...")
    for _, row in df_ranked.iterrows():
        explanation = generate_mmr_explanation(
            icp_description,
            row['short_description'],   # no cleaned text
            model,
            top_n=4
        )
        results.append({
            "company_name": row['company_name'],
            "similarity_score": round(row['similarity_score'], 4),
            "explanation": explanation
        })

    output_file = "icp_ranked_companies_advanced.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to '{output_file}'.\n")

    print("Top 10 Preview:")
    for i, comp in enumerate(results[:10]):
        print(f"{i+1}. {comp['company_name']} | Score: {comp['similarity_score']}")
        print(f"   {comp['explanation']}\n")


# Run engine
if __name__ == "__main__":
    run_icp_engine()
