import pandas as pd
from datasets import load_from_disk, Dataset


def prepare_dataset(dataset, save_path):
    prompts, messages = [], []
    with open("prompts/inference_prompt.txt", "r") as f:
        system_prompt = f.read()
    prompt_template = '''Document:
{}
    '''
    output_template = '''Summary:
{}
    '''
    
    for sample in dataset:
        prompt = prompt_template.format(sample["article"])
        output = output_template.format(sample["highlights"])
        
        message = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': prompt
            },
            {
                'role': 'assistant',
                'content': output
            }
        ]
        prompts.append(prompt)
        messages.append(message)
        
    # saving the data
    df = pd.DataFrame({
        'prompt': prompts,
        'messages': messages
    })
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.01)
    dataset.save_to_disk(save_path)
    print(len(dataset["train"]))
    

if __name__ == "__main__":
    train_subset = load_from_disk("/local3/cui54/summarization_adapter/cnn_dailymail_4000")
    # print(len(train_subset))
    # split_dataset = train_dataset.train_test_split(test_size=0.01)
    # train_split = split_dataset["train"]
    # dev_split = split_dataset["test"]
    # train_split.save_to_disk("/local3/cui54/summarization_adapter/cnn_dailymail_subset/train")
    # dev_split.save_to_disk("/local3/cui54/summarization_adapter/cnn_dailymail_subset/dev")

    prepare_dataset(train_subset, "summarization_data")