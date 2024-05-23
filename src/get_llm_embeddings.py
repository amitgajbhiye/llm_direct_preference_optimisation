from llm2vec import LLM2Vec

import torch
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp")
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised"
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# Encoding documents. Instruction are not required for documents


con_prop_file = "data/ontology_concepts/llama3_with_3inc_exp_con_prop_parsed.txt"
con_prop_df = pd.read_csv(con_prop_file, sep="\t", names=["concept", "property"])

uniq_props = list(con_prop_df["property"].unique())

print(f"Num Unique Property: {len(uniq_props)}")
print(f"Unique Property")
# print(uniq_props)

p_reps = l2v.encode(uniq_props)

print(f"p_reps.shape: {p_reps.shape}")

property_embedding = [(prop, embed) for prop, embed in zip(uniq_props, p_reps)]

print(f"Top 5 props")
print(property_embedding[0:5])

pickle_output_file = "llm_embeds/llama3_with_3inc_exp_prop_embed.pkl"

with open(pickle_output_file, "wb") as pkl_file:
    pickle.dump(property_embedding, pkl_file)
