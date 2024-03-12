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

concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts][0:10]

print(len(concepts), concepts)

basic_prompt = f"What is the most salient property of <CONCEPT>? Generate only the property and do not explain the property."

commonsense_prompt = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately.  
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer, please don't share false information.
<</SYS>>
Write the five most salient properties of the following concept. The propeties must be written in a Python list format. Limit the number of properties to 5. Concept: <CONCEPT>
[/INST]"""

concept_prompts = [commonsense_prompt.replace("<CONCEPT>", con) for con in concepts]

print(concept_prompts)

file_name = "commonsense_prompt_llama2_7b_properties_ufet_concepts.txt"


response_list = []

with open(file_name, "w") as out_file:

    for concept_prompt in concept_prompts:

        sequences = pipeline(
            concept_prompt,
            do_sample=True,
            # top_p=,
            # top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
            max_new_tokens=2000,
            return_full_text=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        for seq in sequences:
            response_list.append(f"{seq['generated_text']}\n\n")
            print(f"{seq['generated_text']}")

            out_file.write(f'{seq["generated_text"]}\n')

            print("===================================")


# parameters = {
#     "max_length": 4000,
#     "max_new_tokens": 1000,
#     "top_k": 10,
#     "return_full_text": False,
#     "do_sample": True,
#     "num_return_sequences": 1,
#     "temperature": 0.8,
#     "repetition_penalty": 1.0,
#     "length_penalty": 1.0,
# }
