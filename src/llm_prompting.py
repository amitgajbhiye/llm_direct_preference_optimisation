import os
import gc
import torch
import transformers

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
)

from peft import PeftModel


# model = "meta-llama/Llama-2-13b-chat-hf"
base_model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-7b-hf"
# model =  "meta-llama/Llama-2-13b-hf"

inp_file = "data/ufet/clean_types.txt"

# Model Prepration

dpo_trained_model = True


if not dpo_trained_model:
# Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


    # Load base moodel in quantised form
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map={"": 0}
    )

    print(f"############ Model ############", end="\n\n")
    print(model, end="\n\n")

else:
    
    adapter = "/home/amit/cardiff_work/llm_direct_preference_optimisation/results/final_checkpoint/"

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapter)

    print (f"Prompting DPO finetuned model")



# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation", model=model, device_map="auto", tokenizer=tokenizer
)


with open(inp_file, "r") as inp_file:
    concepts = inp_file.readlines()

concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

print(f"Number of concepts: {len(concepts)}")


basic_prompt = f"What is the most salient property of <CONCEPT>? Generate only the property and do not explain the property."

commonsense_prompt_1 = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately.  
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer, please don't share false information.
<</SYS>>
Write the five most salient properties of the following concept. The propeties must be written in a Python list format. Limit the number of properties to 5. Concept: <CONCEPT>
[/INST]"""


commonsense_prompt_2 = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. All output must be in valid JSON. Don't add explanation beyond the JSON.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer, please don't share false information.
<</SYS>>
Write the ten most salient properties of the following concept. Output must be in valid JSON like the following example {{"concept": concept, "properties": [in_less_than_ten_words]}}. Output must include only JSON.
All output must be in valid JSON. Don't add any explanations before and after the JSON.
Concept: <CONCEPT>
[/INST]"""


print(f"Prompt used is : {commonsense_prompt_2}")

concept_prompts = [commonsense_prompt_2.replace("<CONCEPT>", con) for con in concepts]

# print(concept_prompts)

file_name = "dpo_finetunned_4bit_cs_prompt2_llama2_7b_properties_ufet_concepts.txt"

# response_list = []

with open(file_name, "w") as out_file:

    for concept_prompt in concept_prompts:

        sequences = pipeline(
            concept_prompt,
            do_sample=True,
            # top_p=,
            # top_k=,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
            max_new_tokens=1000,
            return_full_text=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        for seq in sequences:
            # response_list.append(f"{seq['generated_text']}\n\n")
            print(f"{seq['generated_text']}\n")

            out_file.write(f'{seq["generated_text"]}')

            print("===================================")

        del seq
        del sequences

del model
del pipeline
del concept_prompts


gc.collect()
gc.collect()
gc.collect()
