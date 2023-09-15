import torch
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
import os
import json
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def evaluate(instruction, tokenizer, model, input=None, **kwargs):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda() # these are integers encoded from words
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s) # this will return a fully-wholely description like "Below is an instruction....Response:..."
    return output.split("### Response:")[1].replace("</s>", "").strip()


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

def main():

    # dir_of_hf_w = f"/mnt/swordfish-pool2/asaakyan/style-transfer/alpaca-combo/"
    # out_dir = "../../data/gyafc_w_ICHF_alpaca/formal_to_informal"
    # assert os.path.isdir(out_dir)
    # input_json = "../../data/gyafc_w_ICHF_alpaca/formal_to_informal/test.json"
    # assert os.path.isfile(input_json)
    # model_name = "alpaca_combo"

    dir_of_hf_w = f"/mnt/swordfish-pool2/asaakyan/style-transfer/alpaca-formal-informal-noexpl/"
    out_dir = "../../data/gyafc_w_ICHF_alpaca/formal_to_informal"
    assert os.path.isdir(out_dir)
    input_json = "../../data/gyafc_w_ICHF_alpaca/formal_to_informal/test_noexpl.json"
    assert os.path.isfile(input_json)
    model_name = "alpaca_fif_noexpl"

    load_in_8bit = False
    stop_idx = 10000

    print("Loading the model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(dir_of_hf_w)
    model = LlamaForCausalLM.from_pretrained(
        dir_of_hf_w,
        load_in_8bit= load_in_8bit, # True may save memory (16GB to 10GB), but slower
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    output = []
    with open( input_json,'r') as fp:
        test_file = json.load(fp)
    for i, line in enumerate(test_file):
        if i > stop_idx:
            break
        print(f"INSTANCE {i}, percentage: {i/len(test_file)}")
        print(f"Instruction: {line['instruction']}\nInput: {line['input']}\n")
        pred = evaluate(line['instruction'], tokenizer, model, line['input'])
        output += [{"instruction": line['instruction'],
                "input": line['input'],
                "pred": pred,
                "id": i
                }]
        print("Response:", pred)
        print("-"*50)
        if i%10==0:
            # save list of json with pretty indent
            with open(f"{out_dir}/{model_name}_test_output.json",'w', encoding='utf-8') as fp:
                json.dump(output, fp, indent=4)
    with open(f"{out_dir}/{model_name}_test_output.json",'w', encoding='utf-8') as fp:
                json.dump(output, fp, indent=4)

if __name__ == "__main__":
    main()