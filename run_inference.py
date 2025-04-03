from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import os
import torch
import json
import evaluate
from bert_score import score


def summarize(document, model, tokenizer):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with open('prompts/inference_prompt.txt', 'r') as f:
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
    # print(output)
    output = output.split(document)[1].strip()
    output = output.split("user\nDocument:")[0].strip().split("Summary:")[1].strip()
    return output

def test(dataset, save_path):
    if os.path.exists("summaries.json"):
        with open("summaries.json", "r") as f:
            summaries = json.load(f)
        generated_summaries = summaries["generated_summaries"]
        reference_summaries = summaries["reference_summaries"]
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = "cui54/qwen2.5-3b_b4_t2_LR_1e-5"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        generated_summaries = []
        reference_summaries = []
        for sample in dataset:
            summary = summarize(sample["article"], model, tokenizer)
            generated_summaries.append(summary)
            reference_summaries.append(sample["highlights"])
        summaries = {
            "generated_summaries": generated_summaries,
            "reference_summaries": reference_summaries
        }
        with open(f"summaries/{save_path}.json", "w") as f:
            json.dump(summaries, f, indent=4)
    
    # rouge score
    rouge = evaluate.load("rouge")
    with open(f"eval/{save_path}_rougescore.txt", "w") as file:
        rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
        for key, value in rouge_results.items():
            file.write(f"{key}: {value:.4f}\n")
    
    # bert score
    with open(f"eval/{save_path}_bertscore.txt", "w") as file:
        P, R, F1 = score(generated_summaries, reference_summaries, lang="en", model_type="microsoft/deberta-xlarge-mnli")
        file.write(f"BERTScore precision: {P.mean().item():.4f}\n")
        file.write(f"BERTScore recall: {R.mean().item():.4f}\n")
        file.write(f"BERTScore F1: {F1.mean().item():.4f}")


if __name__ == "__main__":
    test_subset = load_from_disk("/local3/cui54/summarization_adapter/cnn_dailymail_subset/test")
    test(test_subset, "Qwen2.5-3B_b4_t2_1000")