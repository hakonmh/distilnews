import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter


def fix_sentiment_train_data():
    sentiment_files = os.listdir('data/classified-data')
    sentiment_files.remove('aspect_based_analysis.csv')  # Test data

    dfs = []
    for f in sentiment_files:
        df = pd.read_csv('data/classified-data/' + f)
        df = _filter_topic(df)
        if 'Manual Sentiment' in df.columns:
            df['Sentiment'] = df[['Manual Sentiment', 'GPT Sentiment',
                                  'FinRoberta Sentiment']].mode(axis=1)[0]
        else:
            df = df[df['GPT Sentiment'] == df['FinRoberta Sentiment']]
            df['Sentiment'] = df['GPT Sentiment']
        df = df[['Headlines', 'Sentiment']]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['Headlines'] = _clean_text(df['Headlines'])
    df = df.sample(frac=1, random_state=42)
    df = df[df['Headlines'].str.len() > 0].dropna()  # drop empty headlines
    df = df.drop_duplicates(subset='Headlines')

    train, cv = _train_test_split(df, train_size=0.95)
    train = _augment_headlines(train, num_samples=len(train) * 0.33,
                               label_column='Sentiment')

    train.to_csv('data/fixed-data/sentiment-train.csv', index=False)
    cv.to_csv('data/fixed-data/sentiment-val.csv', index=False)
    print('Sentiment train and CV done')


def fix_sentiment_test_data():
    df = pd.read_csv('data/classified-data/aspect_based_analysis.csv')
    df = _filter_topic(df)
    sentiment_columns = ['Manual Sentiment', 'GPT Sentiment', 'FinRoberta Sentiment']
    df['Sentiment'] = df[sentiment_columns].mode(axis=1)[0]
    df = df[['Headlines', 'Sentiment']]
    df['Headlines'] = _clean_text(df['Headlines'])
    df = df.sample(frac=1, random_state=42)
    df.to_csv('data/fixed-data/sentiment-test.csv', index=False)
    print('Sentiment test done')


def fix_topic_data():
    topic_files = ['ag-news-classification-dataset.csv', 'headlines-5000.csv',
                   'labelled_newscatcher_dataset.csv', 'News_Category_Dataset_v3.csv',
                   'title_and_headline_sentiment_prediction.csv', 'twitter-financial-news.csv']
    dfs = []
    for f in topic_files:
        df = pd.read_csv('data/classified-data/' + f)
        df['Topic'] = df[['GPT Topic', 'CardiffNLP Topic', 'Topic_04 Topic']].mode(axis=1)[0]
        df = df[['Headlines', 'Topic']]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['Headlines'] = _clean_text(df['Headlines'])
    df = df.sample(frac=1, random_state=42)
    df = df[df['Headlines'].str.len() > 0].dropna()  # drop empty headlines
    df = df.drop_duplicates(subset='Headlines')

    train, test = _train_test_split(df, train_size=0.95)
    train, cv = _train_test_split(train, train_size=0.95)
    train = _augment_headlines(train, num_samples=len(train) * 0.33,
                               label_column='Topic')

    train.to_csv('data/fixed-data/topic-train.csv', index=False)
    cv.to_csv('data/fixed-data/topic-val.csv', index=False)
    test.to_csv('data/fixed-data/topic-test.csv', index=False)
    print('Topic train, CV, and test done')


def _train_test_split(df, train_size=0.95):
    df = df.sample(frac=1, random_state=42)
    train, test = np.split(df, [int(train_size * len(df))])
    return train, test


def _filter_topic(df):
    """Filter out non-economics topics."""
    topic = df[['GPT Topic', 'CardiffNLP Topic', 'Topic_04 Topic']].mode(axis=1)[0]
    mask = topic == 'Economics'
    df = df[mask]
    return df


def _clean_text(headlines: pd.Series):
    """Clean headlines function

    This function takes a pandas Series containing text headlines as input and performs
    the following cleaning operations:

    1. Converts all text to lowercase.
    2. Replaces all non-alphanumeric characters except for ".", ",", "!", "?", and "-" with
    a space.
    3. Replaces multiple consecutive spaces with a single space.
    4. Strips any leading or trailing whitespace from each headline.
    5. Removes the string '- analyst blog' from each headline.

    Parameters
    ----------
    headlines : pandas.Series
        A pandas Series containing text headlines to be cleaned.

    Returns
    -------
    cleaned_headlines : pandas.Series
        A pandas Series containing the cleaned text headlines.
    """
    headlines = headlines.astype(str)
    headlines = headlines.str.lower()
    headlines = headlines.str.replace(r'[^a-zA-Z0-9.,?!-]', ' ', regex=True)
    headlines = headlines.str.replace(r'\s+', ' ', regex=True)
    headlines = headlines.str.replace('- analyst blog', '')
    headlines = headlines.str.strip()
    return headlines


def _augment_headlines(df, num_samples, label_column):
    """Augment headlines.

    This function takes a pandas DataFrame of headlines and their sentiment labels,
    and returns a new DataFrame with augmented headlines. The function uses three
    different augmentation methods to generate new headlines from the original ones.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing two columns: 'Headlines', which contains the original headlines,
        and label_column, which contains the labels for each headline.
    num_samples : int
        The number of augmented samples to generate for each original headline.
    label_column : str
        The name of the column in df that contains the labels for each headline.

    Returns
    -------
    df : pandas.DataFrame
        A new DataFrame containing the original headlines and the augmented headlines.
        The DataFrame has two columns: 'Headlines', which contains the original and augmented
        headlines, and label_column, which contains the corresponding labels.
    """
    tasks = _create_augmentation_tasks(df, num_samples, label_column)
    with Pool(max(cpu_count() - 1, 1)) as pool:
        results = pool.map(_augment_headline, tasks)

    augmented_sentences = [result[0] for result in results if result is not None]
    augmented_labels = [result[1] for result in results if result is not None]

    new_df = pd.DataFrame({'Headlines': augmented_sentences, label_column: augmented_labels})
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42)
    df = df.drop_duplicates(subset='Headlines')
    df = df[df['Headlines'].str.len() > 0].dropna()  # drop empty headlines
    return df


def _create_augmentation_tasks(df, num_samples, label_column):
    methods = [WordNetAugmenter(), EmbeddingAugmenter(), EasyDataAugmenter()]
    tasks = []
    for _ in range(int(num_samples)):
        index = random.randint(0, len(df) - 1)
        headline = df['Headlines'].iloc[index]
        label = df[label_column].iloc[index]

        # randomly select an augmentation method
        method_id = random.randint(0, len(methods) - 1)
        augmenter = methods[method_id]

        tasks.append((headline, label, augmenter))
    return tasks


def _augment_headline(args):
    original_sentence, label, augmenter = args
    try:
        augmented_sentence = augmenter.augment(original_sentence)
        return (augmented_sentence[0], label)
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    if not os.path.exists('data/fixed-data'):
        os.makedirs('data/fixed-data')
    fix_sentiment_train_data()
    fix_sentiment_test_data()
    fix_topic_data()
