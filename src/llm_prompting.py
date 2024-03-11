import gc
import torch
import transformers
from transformers import AutoTokenizer

torch.cuda.empty_cache()
gc.collect()
gc.collect()


# model = "meta-llama/Llama-2-13b-chat-hf"
# model = "meta-llama/Llama-2-7b-chat-hf"
model = "meta-llama/Llama-2-7b-hf"
# model =  "meta-llama/Llama-2-13b-hf"

inp_file = "data/ufet/clean_types.txt"


tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


with open(inp_file, "r") as inp_file:
    concepts = inp_file.readlines()

concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts][0:20]

print(len(concepts), concepts)

prompt = f"What is the most salient property of <CONCEPT>? Generate only the property and do not explain the property."
concept_prompts = [prompt.replace("<CONCEPT>", con) for con in concepts][0:150]

print(concept_prompts)

file_name = "individual_property_llama2_7b_properties_mcrae_concepts.txt"


response_list = []

with open(file_name, "w") as out_file:

    for concept_prompt in concept_prompts:
        # print (concept_prompt)

        for i in range(0, 11):

            print(f"i : {i} : {concept_prompt}")

            sequences = pipeline(
                concept_prompt,
                do_sample=True,
                top_k=5,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=200,
            )

            for seq in sequences:
                response_list.append(f"{seq['generated_text']}\n\n")
                print(f"{seq['generated_text'][len(concept_prompt):]}")

                out_file.write(f'{seq["generated_text"][len(concept_prompt):]}\n')

                print("===================================")
