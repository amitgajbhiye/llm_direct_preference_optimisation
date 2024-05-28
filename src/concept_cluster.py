import pickle

import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# Loading MNTP (Masked Next Token Prediction) model.
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
)
# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)


# Loading ontology concepts
ontology_file = "data/ontology_concepts/transport_vocab.txt"

with open(ontology_file, "r") as fin:
    concepts = [con.strip("\n") for con in fin.readlines()]

print(f"num_concepts: {len(concepts)}")

llm_con_embeds = l2v.encode(concepts).detach().cpu().numpy()

print(f"llm_con_embeds.shape: {llm_con_embeds.shape}")

cons_and_embedding = [(prop, embed) for prop, embed in zip(concepts, llm_con_embeds)]

print(f"Top 5 props")
print(cons_and_embedding[0:5])

pickle_output_file = "output_llm_embeds/transport_concepts.pkl"
with open(pickle_output_file, "wb") as pkl_file:
    pickle.dump(cons_and_embedding, pkl_file)


# Normalize the embeddings
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(llm_con_embeds)

eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples = [1, 3, 5, 7, 9, 10]

for ep in eps_range:
    for min in min_samples:

        dbscan = DBSCAN(eps=ep, min_samples=min, metric="cosine", algorithm="brute")
        clusters = dbscan.fit_predict(embeddings_normalized)

        concept_cluster_list = [
            (con, clus_label) for con, clus_label in zip(concepts, clusters)
        ]

        con_cluster_df = pd.DataFrame.from_records(
            concept_cluster_list, columns=["concept", "cluster_label"]
        )

        # property_cluster_df = pd.DataFrame.from_dict(property_cluster_map)
        con_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

        print(f"Eps: {ep}, Min_sam: {min}")
        print(f"con_cluster_df")
        print(con_cluster_df)

        con_cluster_df.to_csv(
            f"data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/eps{ep}_minsam{min}_transport_con_cluster_llama38b_embeds.txt",
            sep="\t",
            index=None,
        )
