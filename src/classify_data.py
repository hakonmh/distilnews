"""
Classifies the different datasets used using OpenAI's GPT-3.5 model.
"""
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import torch
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from _pipelines import topic_pipelines, sentiment_pipelines

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
openai.api_key = os.environ["OPENAI_API_KEY"]

SYSTEM_PROMPT = """You are a world-class news headline topic and sentiment classifier \
for a financial hedge fund. Your job is to classify news headlines based on:
- Topic (Economics or Other).
- Bullish/Bearish Sentiment (Positive, Neutral, or Negative) as it relates to the future \
prospects of the economy.

Your answer should be in the following format:
num. Topic, Sentiment

For example:
1. Economics, Positive
2. Other, Neutral
"""


def process_data_pipeline(pipeline, file_path):
    """Process data pipeline function

    This function takes a data processing pipeline and a file path as input,
    applies the pipeline to the file and saves the resulting data to a CSV file.
    It returns the total number of tokens in the processed data..

    Args:
        pipeline (function): A function that takes a file path and returns a pandas DataFrame.
        file_path (str): The path to the input file.
        cols (list): A list of column names to include in the output file.

    Returns:
        int: The total number of tokens processed.
    """
    file_name = os.path.basename(file_path)
    file_name = re.sub(r"\.\w+$", ".csv", file_name)
    df = pipeline(file_path)
    df = df.reset_index(drop=True)
    cols = df.columns.tolist()
    cols.extend(["Topic", "GPT Sentiment", "FinBERT Sentiment"])

    if not os.path.exists('./data/classified-data'):
        os.makedirs('./data/classified-data')
    output_file_path = os.path.join(r".\data\classified-data", file_name)
    _touch_file(output_file_path, cols=cols)

    chunk_size = 20
    total_tokens = 0
    for chunk, tokens in _process_chunks(df, chunk_size):
        chunk.to_csv(output_file_path, mode='a', header=False, index=False)
        total_tokens += tokens
    return total_tokens


def _touch_file(file_path, cols):
    """Create a new file with the given columns"""
    str_cols = ",".join(cols)
    with open(file_path, 'w', newline='') as f:
        f.write(f"{str_cols}\n")


def _process_chunks(df, chunk_size):
    chunks = _split_df(df, chunk_size)
    model, tokenizer = _launch_finbert()
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_process_chunk, chunk, model, tokenizer) for chunk in chunks}
        for future in as_completed(futures):
            yield future.result()


def _split_df(df, chunk_size):
    return np.array_split(df, len(df) // chunk_size + 1)


def _launch_finbert():
    model = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone", num_labels=3
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return model, tokenizer


def _process_chunk(chunk, model, tokenizer):
    chunk = chunk.reset_index(drop=True)
    headlines = chunk['Headlines'].tolist()
    gpt_output, tokens = _classify_headlines_gpt(headlines)
    topic, gpt_sentiment = _parse_gpt_output(gpt_output)
    finbert_sentiment = _classify_headlines_finbert(headlines, model, tokenizer)
    try:
        df = pd.DataFrame({
            'Topic': topic,
            'GPT Sentiment': gpt_sentiment,
            'FinBERT Sentiment': finbert_sentiment}
        )
        df = pd.concat([chunk, df], axis=1, ignore_index=True)
    except Exception as e:
        print(repr(e))
        df = None
    return df, tokens


def _classify_headlines_finbert(headlines, model, tokenizer):
    headlines = tokenizer(headlines, add_special_tokens=True, max_length=30,
                          padding="max_length", truncation=True, return_tensors="pt")
    headlines = headlines.to(device=device)
    output = model(**headlines)

    probs = torch.nn.functional.softmax(output.logits, dim=1)
    preds = probs.argmax(dim=1).to('cpu').numpy()
    preds = pd.Series(preds).map({0: "Neutral", 1: "Positive", 2: "Negative"})
    return preds


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _classify_headlines_gpt(headlines):
    """Classify the topic and sentiment of news headlines using GPT-3.5"""
    user_prompt = "Classify the following headlines:"
    for idx, headline in enumerate(headlines):
        user_prompt += f"\n{idx + 1}. {headline}"

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=300,
        temperature=0.2,
        frequency_penalty=-0.1,
        presence_penalty=-0.1
    )
    return response['choices'][0]['message']['content'], response['usage']['total_tokens']


def _parse_gpt_output(text):
    headlines = text.split("\n")
    try:
        topics = [h.split(",")[0].split(".")[1].strip() for h in headlines]
        sentiment = [h.split(",")[1].strip() for h in headlines]
    except IndexError:
        print(f'IndexError: {text}')
        topics = ["Neither"] * len(headlines)
        sentiment = ["Neutral"] * len(headlines)

    for idx, c in enumerate(topics):
        if 'Economy' in c:
            topics[idx] = 'Economics'
        if c not in {'Economics', 'Other'}:
            print(f'Topic: {c}')
            topics[idx] = 'Other'

    for idx, s in enumerate(sentiment):
        if s == "Bullish":
            sentiment[idx] = "Positive"
        elif s == "Bearish":
            sentiment[idx] = "Negative"
        elif s == "Neither":
            sentiment[idx] = "Neutral"
        elif s not in ["Positive", "Negative", "Neutral"]:
            print(f'Sentiment: {s}')
            sentiment[idx] = "Neutral"

    return topics, sentiment


if __name__ == '__main__':
    total_tokens = 0
    pipelines = {**topic_pipelines, **sentiment_pipelines}
    for name, pipeline in pipelines.items():
        total_tokens += process_data_pipeline(
            pipeline['pipeline'],
            pipeline['file_path'],
            pipeline['cols']
        )
        print(f"{name} done!")
    print(f"Total tokens: {total_tokens}")
