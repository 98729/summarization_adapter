import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from datasets import load_dataset, load_from_disk
import evaluate
from bert_score import score
import os
import wandb

def load_data():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset

def generate_summary(document, zero_shot=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    if zero_shot:
        with open("prompts/zeroshot_prompt.json", "r") as f:
            instruction = json.load(f)
        template = instruction["template"]
        input_text = template.format(document=document)
        messages = [{"role": "user", "content": input_text}]
    else:
        with open("prompts/fewshot_prompt.json", "r") as f:
            instruction = json.load(f)
        intro = instruction["intro"]
        template = instruction["template"]
        examples = instruction["examples"]
        input_text = template.format(document=document)
        messages = [{"role": "system", "content": intro}]
        
        for example in examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})
        messages.append({"role": "user", "content": input_text})
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.6)
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    outputs = output.split(document)

    summary = outputs[-1].strip()
    if not zero_shot and "assistant" in summary:
        summary = summary.split("assistant", 1)[1].strip()
    # print(output)
    return summary

def construct_prompt(document, instruction):
    intro = instruction["intro"]
    template = instruction["template"]
    examples = instruction["examples"]

    messages = [{"role": "system", "content": intro}]
    for example in examples:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    input_text = template.format(document=document)
    messages.append({"role": "user", "content": input_text})

    return messages

def prepare_data(dataset, tokenizer, instruction):
    tokenized_data = []

    for sample in dataset:
        messages = construct_prompt(sample["article"], instruction)
        
        # Append summary as assistant message to align input/labels
        messages.append({"role": "assistant", "content": sample["highlights"]})

        # Tokenize the full sequence (including summary)
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").squeeze(0)

        # Create label IDs by copying input_ids
        labels = input_ids.clone()

        # Find the start position of the assistant response
        summary_start_idx = len(tokenizer.apply_chat_template(messages[:-1], return_tensors="pt").squeeze(0))

        # Mask out non-summary parts (-100 to ignore in loss computation)
        labels[:summary_start_idx] = -100  

        tokenized_data.append({"input_ids": input_ids, "labels": labels})

    return tokenized_data

def train(dataset, save_path):
    wandb.init(project="qwen2.5_3b_summarize_finetuning")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-3B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    with open("prompts/prompt.json", "r") as f:
        instruction = json.load(f)
    train_data = prepare_data(dataset, tokenizer, instruction)

    train_args = TrainingArguments(
        output_dir=f"{save_path}_checkpoints",
        per_device_train_batch_size=4,  # Adjust batch size based on GPU memory
        num_train_epochs=2,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,
        report_to="wandb",
        save_steps=100,
        save_total_limit=2
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        data_collator=data_collator
    )
    train_dataloader = trainer.get_train_dataloader()
    for batch in train_dataloader:
        print(f"Logits shape: {batch['input_ids'].shape}")  # Should be (batch_size, seq_length)
        print(f"Labels shape: {batch['labels'].shape}")  # Should be (batch_size, seq_length)
        break
    trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def test(dataset, metric, save_path, num_samples=1, zeroshot=False):
    if os.path.exists(f"summaries/{save_path}.json"):
        with open(f"summaries/{save_path}.json", "r") as f:
            summaries = json.load(f)
        generated_summaries = summaries["generated_summaries"]
        reference_summaries = summaries["reference_summaries"]
    else:
        generated_summaries = []
        reference_summaries = []
        for sample in dataset:
            summary = generate_summary(sample["article"], zero_shot=zeroshot)
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
    # download and save subset of cnn_dailymail
    # dataset = load_data()
    # train_subset = dataset["train"].select(range(8000))
    # test_subset = dataset["test"].select(range(100))
    # train_subset.save_to_disk("/local3/cui54/summarization_adapter/cnn_dailymail_8000")
    # test_subset.save_to_disk("/local3/cui54/summarization_adapter/cnn_dailymail_subset/test")
    
    # load cnn_dailymail
    # train_subset = load_from_disk("/local3/cui54/summarization_adapter/cnn_dailymail_subset/train")
    test_subset = load_from_disk("/local3/cui54/summarization_adapter/cnn_dailymail_test")
    # test(test_subset, "zeroshot", zeroshot=True)
    # test(test_subset, "zeroshot", zeroshot=True)
    test(test_subset, "fewshot")
    test(test_subset, "fewshot")

    # train(train_subset, "Qwen2.5-3B_b4_t2")