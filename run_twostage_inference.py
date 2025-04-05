from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import os
import torch
import json
import evaluate
from bert_score import score

def extract(document, model, tokenizer):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with open('prompts/extract_inference_prompt.txt', 'r') as f:
        system_prompt = f.read()
    prompt_template = '''Document:
{}
    '''
    prompt = prompt_template.format(document)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.6)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # output = output.split(document)[1].strip()
    print(output)
    # if "Document:" in output:
    #     output = output.split("Document:")[0].strip()
    # if "Summary:" in output:
    #     output = output.split("Summary:")[1].strip()
    return output


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "cui54/qwen2.5-3b_b4_t2_LR_1e-5_1000_augment_stage2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    test_subset = load_from_disk("/local3/cui54/summarization_adapter/cnn_dailymail_test")
    
    for sample in test_subset:
        extract(sample, model, tokenizer)
        break