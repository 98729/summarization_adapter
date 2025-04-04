import openai
import os
from openai import OpenAI
from datasets import load_from_disk
from tqdm import tqdm

def call_llm(prompt, sys_prompt, model='gpt-4o-mini', stop=None, return_json=False, max_tokens=None, temperature=0.5):
    client = OpenAI(
        organization=os.environ['OPENAI_ORG_ID'],
        api_key=os.environ['OPENAI_API_KEY']
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" } if return_json else openai.NOT_GIVEN,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature
    )
    return completion.choices[0].message.content

def extract(document, summary):
    with open("prompts/extract_prompt.txt", "r") as f:
        system_prompt = f.read()
    prompt_template = '''Document:
{}

Summary:
{}

Key Phrases:\n
    '''
    prompt = prompt_template.format(document, summary)
    response = call_llm(prompt, system_prompt).strip()
    # print(document)
    # print(summary)
    # print(response)
    return response

def add_phrases(datapath):
    dataset = load_from_disk(f"/local3/cui54/summarization_adapter/{datapath}")

    def process(sample):
        sample["key phrases"] = extract(sample["article"], sample["highlights"])
        return sample

    dataset = dataset.map(process, desc="Processing Samples")
    dataset.save_to_disk(f"/local3/cui54/summarization_adapter/{datapath}_augment")


if __name__ == "__main__":
    add_phrases("cnn_dailymail_1000")