import pandas as pd
import os
import re
import csv
import numpy as np


def ag_news_classification_pipeline(file_path):
    """For: https://www.kaggle.com/amananandrai/ag-news-classification-dataset"""
    df = _read_csv_with_split(file_path)
    df = df[['Title', 'Class Index']]
    df.columns = ['Headlines', 'Manual Topic']

    MAPPING = {'1': 'World', '2': 'Sports', '3': 'Business', '4': 'Tech'}
    df.loc[:, 'Manual Topic'] = df['Manual Topic'].map(MAPPING)

    df.loc[:, 'Headlines'] = _clean_headlines(df['Headlines'])
    # Remove any text enclosed in parentheses
    df.loc[:, 'Headlines'] = df['Headlines'].str.replace(
        r'\([^)]*\)', '', regex=True
    )
    # remove  &lt;b&gt;...&lt;/b&gt; from text
    df.loc[:, 'Headlines'] = df['Headlines'].str.replace(
        r"&lt;b&gt;.*?&lt;/b&gt;", "", regex=True
    )
    return df.dropna()


def _read_csv_with_split(filename):
    """Reads a CSV file and splits each line by the first three commas and combines the rest."""
    rows = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > 3:
                row[2] = ','.join(row[2:])
                row = row[:3]
            rows.append(row)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df


def headlines_5000_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/lazrus/headlines-5000"""
    df = pd.read_csv(file_path)
    df = df[['headline', 'category']]
    df.columns = ['Headlines', 'Manual Topic']
    df.loc[:, 'Headlines'] = _clean_headlines(df['Headlines'])
    df.loc[:, 'Manual Topic'] = 'Economics'
    return df.dropna()


def newscatcher_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset"""
    df = pd.read_csv(file_path, sep=';')
    df = df[['published_date', 'title', 'topic']]
    df.columns = ['Time', 'Headlines', 'Manual Topic']
    df.loc[:, 'Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by='Time')
    df = df[['Headlines', 'Manual Topic']]
    df.loc[:, 'Headlines'] = _clean_headlines(df['Headlines'])
    return df.dropna()


def news_category_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/rmisra/news-category-dataset"""
    df = pd.read_json(file_path, lines=True)
    df = df[['headline', 'category']]
    df.columns = ['Headlines', 'Manual Topic']
    df.loc[:, 'Headlines'] = _clean_headlines(df['Headlines'])
    df.loc[:, 'Headlines'] = df['Headlines'].apply(_remove_ambigiuous_characters)
    return df.dropna()


def _remove_ambigiuous_characters(text):
    # Replace non-breaking space with a regular space
    text = text.replace(u'\xa0', u' ')
    # Remove any other non-printable or non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def title_and_headline_sentiment_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/adarshsng/title-and-headline-sentiment-prediction"""
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df[['Title', 'Topic']]
    df.columns = ['Headlines', 'Manual Topic']
    df.loc[:, 'Headlines'] = _clean_headlines(df['Headlines'])
    # Drop titles that ends with "..."
    df = df[~df['Headlines'].str.endswith('...')]
    df.loc[:, 'Manual Topic'] = np.where(df['Manual Topic'] == 'economy', 'Economics', 'Other')
    return df.dropna()


def twitter_financial_news_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/sulphatet/twitter-financial-news"""
    df = pd.read_csv(file_path)
    df = df[['text', 'label']]
    df.columns = ['Headlines', 'Manual Topic']

    df.loc[:, 'Headlines'] = _clean_headlines(df['Headlines'])
    # Remove everything in headlines after https://t.co/
    df.loc[:, 'Headlines'] = df['Headlines'].apply(lambda x: re.sub(r' https://t.co/.*', '', x))
    df.loc[:, 'Manual Topic'] = 'Economics'
    return df.dropna()


def aspect_pipeline(file_path):
    """
    For: https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news
    """
    df = pd.read_csv(file_path)
    df = df[['Title', 'Decisions']]
    df.columns = ['Headlines', 'Manual Sentiment']
    df['Manual Sentiment'] = df['Manual Sentiment'].str.split('"').str[3]
    df['Manual Sentiment'] = df['Manual Sentiment'].str.capitalize()
    df['Headlines'] = _clean_headlines(df['Headlines'])
    return df.dropna()


def cnbc_guardian_reuters_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/notlucasp/financial-news-headlines"""
    df = pd.read_csv(file_path)
    df = df['Headlines'].to_frame()
    df['Headlines'] = _clean_headlines(df['Headlines'])
    return df.dropna()


def financial_phrasebank_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news"""
    df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
    df.columns = ['Manual Sentiment', 'Headlines']
    df = df[['Headlines', 'Manual Sentiment']]
    df['Manual Sentiment'] = df['Manual Sentiment'].str.capitalize()
    df['Headlines'] = _clean_headlines(df['Headlines'])
    return df


def gold_dataset_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold"""
    df = pd.read_csv(file_path)
    df = df[['News', 'Price Sentiment']]
    df.columns = ['Headlines', 'Manual Sentiment']
    df['Headlines'] = _clean_headlines(df['Headlines'])
    df['Manual Sentiment'] = df['Manual Sentiment'].str.capitalize()
    df.loc[df['Manual Sentiment'] == 'None', 'Manual Sentiment'] = 'Neutral'
    return df.dropna()


def kaggle_financial_sentiment_pipeline(file_path):
    """For: https://huggingface.co/datasets/chiapudding/kaggle-financial-sentiment"""
    df = pd.read_csv(file_path)
    df.columns = ['Headlines', 'Manual Sentiment']
    df['Manual Sentiment'] = df['Manual Sentiment'].str.capitalize()
    df['Headlines'] = _clean_headlines(df['Headlines'])
    return df.dropna()


def raw_partner_pipeline(file_path):
    """
    For: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
    """
    df = pd.read_csv(file_path)
    df = df['headline'].to_frame()
    df.columns = ['Headlines']
    df['Headlines'] = _clean_headlines(df['Headlines'])
    df = df.drop_duplicates()
    df = df.sample(n=300000, random_state=42, replace=False)
    return df.dropna()


def reuters_old_news_pipeline(main_folder):
    """For: https://github.com/duynht/financial-news-dataset"""
    headlines = []
    for folder in os.listdir(main_folder):
        if not os.path.isdir(os.path.join(main_folder, folder)):
            continue
        for file_name in os.listdir(os.path.join(main_folder, folder)):
            file_path = os.path.join(main_folder, folder, file_name)
            # Read first line
            with open(file_path, 'r') as f:
                headline = f.readline()
            headlines.append(headline.strip('-- ').strip('\n'))
    df = pd.Series(headlines)
    df = _clean_headlines(df)
    df = df.to_frame()
    df.columns = ['Headlines']
    return df.dropna()


def stock_news_pipeline(file_path):
    """For: https://www.kaggle.com/datasets/johoetter/labeled-stock-news-headlines"""
    df = pd.read_csv(file_path)
    df = df[['headline', 'label']]
    df.columns = ['Headlines', 'Manual Sentiment']
    df['Headlines'] = _clean_headlines(df['Headlines'])
    return df.dropna()


def _clean_headlines(headlines: pd.Series):
    """Removes duplicate spaces, leading/trailing whitespace and spaces before punctuation.
    quotes.
    """
    headlines = headlines.copy()
    # Removes emojis
    headlines = headlines.str.encode('ascii', 'ignore').str.decode('ascii')
    # Remove spaces before punctuation
    headlines = headlines.str.replace(r"\s([?.!,:;])", r"\1", regex=True)
    # Remove duplicate spaces
    headlines = headlines.str.replace(r"\s+", " ", regex=True)
    headlines = headlines.str.strip()
    return headlines


topic_pipelines = {
    'ag-news-classification': {
        'pipeline': ag_news_classification_pipeline,
        'file_path': r"./data/raw-topic/ag-news-classification-dataset.csv",
    },
    'headlines-5000': {
        'pipeline': headlines_5000_pipeline,
        'file_path': r"./data/raw-topic/headlines-5000.csv",
    },
    'labelled_newscatcher': {
        'pipeline': newscatcher_pipeline,
        'file_path': r"./data/raw-topic/labelled_newscatcher_dataset.csv",
    },
    'news_category': {
        'pipeline': news_category_pipeline,
        'file_path': r"./data/raw-topic/News_Category_Dataset_v3.json",
    },
    'title_and_headline_sentiment': {
        'pipeline': title_and_headline_sentiment_pipeline,
        'file_path': r"./data/raw-topic/title_and_headline_sentiment_prediction.csv",
    },
    'twitter-financial-news': {
        'pipeline': twitter_financial_news_pipeline,
        'file_path': r"./data/raw-topic/twitter-financial-news.csv",
    },
}

sentiment_pipelines = {
    'aspect_based_analysis': {
        'pipeline': aspect_pipeline,
        'file_path': r"./data/raw-sentiment/aspect_based_analysis.csv",
    },
    'cnbc_headlines': {
        'pipeline': cnbc_guardian_reuters_pipeline,
        'file_path': r"./data/raw-sentiment/cnbc_headlines.csv",
    },
    'financial_phrasebank': {
        'pipeline': financial_phrasebank_pipeline,
        'file_path': r"./data/raw-sentiment/financial_phrasebank.csv",
    },
    'gold_dataset': {
        'pipeline': gold_dataset_pipeline,
        'file_path': r"./data/raw-sentiment/gold-dataset-sinha-khandait.csv",
    },
    'guardian_headlines': {
        'pipeline': cnbc_guardian_reuters_pipeline,
        'file_path': r"./data/raw-sentiment/guardian_headlines.csv",
    },
    'kaggle_financial_sentiment': {
        'pipeline': kaggle_financial_sentiment_pipeline,
        'file_path': r"./data/raw-sentiment/kaggle_financial_sentiment.csv",
    },
    'raw_partner_headlines': {
        'pipeline': raw_partner_pipeline,
        'file_path': r"./data/raw-sentiment/raw_partner_headlines.csv",
    },
    'ReutersNews': {
        'pipeline': reuters_old_news_pipeline,
        'file_path': r"./data/raw-sentiment/ReutersNews106521",
    },
    'reuters_headlines': {
        'pipeline': cnbc_guardian_reuters_pipeline,
        'file_path': r"./data/raw-sentiment/reuters_headlines.csv",
    },
    'stock_news': {
        'pipeline': stock_news_pipeline,
        'file_path': r"./data/raw-sentiment/stock_news.csv",
    },
}
