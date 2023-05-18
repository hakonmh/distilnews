import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter


def fix_sentiment_data():
    sentiment_files = os.listdir('data/classified-data')
    dfs = []
    for f in sentiment_files:
        df = pd.read_csv('data/classified-data/' + f)
        df = _process_and_label_df(df, filter_topic=True)
        dfs.append(df[['Headlines', 'Sentiment']])

    df = pd.concat(dfs, ignore_index=True)
    df = _clean_data(df)
    train, val, test = _train_val_test_split(df, train_size=0.9)
    train = _augment_headlines(train, num_samples=len(train) * 0.33)

    train.to_csv('data/fixed-data/sentiment-train.csv', index=False)
    val.to_csv('data/fixed-data/sentiment-val.csv', index=False)
    test.to_csv('data/fixed-data/sentiment-test.csv', index=False)
    print('Sentiment train, test and CV done')


def fix_topic_data():
    topic_files = ['ag-news-classification-dataset.csv', 'headlines-5000.csv',
                   'labelled_newscatcher_dataset.csv', 'News_Category_Dataset_v3.csv',
                   'ReutersNews106521.csv', 'reuters_headlines.csv', 'guardian_headlines.csv',
                   'title_and_headline_sentiment_prediction.csv', 'twitter-financial-news.csv']
    dfs = []
    for f in topic_files:
        df = pd.read_csv('data/classified-data/' + f)
        df = _process_and_label_df(df, filter_topic=False)
        dfs.append(df[['Headlines', 'Topic']])

    df = pd.concat(dfs, ignore_index=True)
    df = _clean_data(df)
    train, val, test = _train_val_test_split(df, train_size=0.9)
    train = _augment_headlines(train, num_samples=len(train) * 0.33)

    train.to_csv('data/fixed-data/topic-train.csv', index=False)
    val.to_csv('data/fixed-data/topic-val.csv', index=False)
    test.to_csv('data/fixed-data/topic-test.csv', index=False)
    print('Topic train, CV, and test done')


def _process_and_label_df(df, filter_topic=False):
    """Process a dataframe and labels the Sentiment and Topic with the majority vote of the
    classifiers available."""
    df['Headlines'] = __clean_text(df['Headlines'])
    df['Manual Topic'] = __fix_manual_topic(df)
    df['Topic'] = __label_topic(df)
    df['Sentiment'] = __label_sentiment(df)
    if filter_topic:
        df = df[df['Topic'] == 'Economics']
    return df[['Headlines', 'Sentiment', 'Topic']]


def __clean_text(headlines: pd.Series):
    """This function takes a pandas Series containing text headlines as input and performs
    the following cleaning operations:

    1. Converts all text to lowercase.
    2. Replaces all non-alphanumeric characters except for ".", ",", "!", "?", and "-" with
    a space.
    3. Replaces multiple consecutive spaces with a single space.
    4. Strips any leading or trailing whitespace from each headline.
    5. Removes the string '- analyst blog' from each headline.
    """
    headlines = headlines.astype(str)
    headlines = headlines.str.lower()
    headlines = headlines.str.replace(r'[^a-zA-Z0-9.,?!-]', ' ', regex=True)
    headlines = headlines.str.replace(r'\s+', ' ', regex=True)
    headlines = headlines.str.replace('- analyst blog', '')
    headlines = headlines.str.strip()
    return headlines


def __fix_manual_topic(df):
    """Converts the human labelled topic to either Economics or Other."""
    MANUAL_TOPICS = ['BUSINESS', 'MONEY', 'TECH', 'TECHNOLOGY', 'ECONOMY', 'ECONOMICS']
    if "Manual Topic" in df.columns:
        manual_topic = df['Manual Topic']
    else:
        manual_topic = pd.Series(['BUSINESS'] * len(df))
    manual_topic = manual_topic.str.upper()
    manual_topic = np.where(manual_topic.isin(MANUAL_TOPICS), 'Economics', 'Other')
    return manual_topic


def __label_topic(df):
    """Labels the topic with the majority vote of the classifiers available."""
    topic = df[['GPT Topic', 'CardiffNLP Topic', 'Topic_04 Topic']].mode(axis=1)[0]
    is_economics = (topic == 'Economics') & (df['Manual Topic'] == 'Economics')
    topic = np.where(is_economics, 'Economics', 'Other')
    return topic


def __label_sentiment(df):
    """Labels the sentiment with the majority vote of the classifiers available, drops rows
    where no mode is available."""
    if 'Manual Sentiment' in df.columns:
        sentiment = df[['Manual Sentiment', 'GPT Sentiment', 'FinRoberta Sentiment']].mode(axis=1)[0]
    else:
        sentiment = df[df['GPT Sentiment'] == df['FinRoberta Sentiment']]
        sentiment = sentiment['GPT Sentiment']
    return sentiment


def _clean_data(df):
    """Cleans the data by removing empty headlines and duplicates."""
    df = df.drop_duplicates(subset='Headlines')
    df = df[df['Headlines'].str.len() > 0].dropna()  # drop empty headlines
    df = df[df['Headlines'].str.lower() != 'nan']  # drop headlines that are 'nan'
    return df.sample(frac=1, random_state=42, ignore_index=True)


def _train_val_test_split(df, train_size=0.90):
    """Splits the data into train, validation, and test sets where the test and
    validation set sizes are each equal to (1-train_size) / 2."""
    label_column = df.columns.difference(['Headlines'])[0]
    # First, split df into 90% train and 10% val_test
    train, val_test = train_test_split(df, train_size=train_size, stratify=df[label_column], random_state=42)
    # Then, split val_test into validation and test sets
    val, test = train_test_split(val_test, test_size=0.5, stratify=val_test[label_column], random_state=42)
    return train, val, test


def _augment_headlines(df, num_samples):
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

    Returns
    -------
    df : pandas.DataFrame
        A new DataFrame containing the original headlines and the augmented headlines.
        The DataFrame has two columns: 'Headlines', which contains the original and augmented
        headlines, and label_column, which contains the corresponding labels.
    """
    label_column = df.columns.difference(['Headlines'])[0]

    tasks = __create_augmentation_tasks(df, num_samples, label_column)
    with Pool(max(cpu_count() - 1, 1)) as pool:
        results = pool.map(__augment_headline, tasks)

    augmented_sentences = [result[0] for result in results if result is not None]
    augmented_labels = [result[1] for result in results if result is not None]

    new_df = pd.DataFrame({'Headlines': augmented_sentences, label_column: augmented_labels})
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42, ignore_index=True)
    df = df.drop_duplicates(subset='Headlines')
    df = df[df['Headlines'].str.len() > 0].dropna()  # drop empty headlines
    return df


def __create_augmentation_tasks(df, num_samples, label_column):
    agumenters = [WordNetAugmenter(), EmbeddingAugmenter(), EasyDataAugmenter()]
    tasks = []
    for _ in range(int(num_samples)):
        index = random.randint(0, len(df) - 1)
        headline = df['Headlines'].iloc[index]
        label = df[label_column].iloc[index]

        # randomly select an augmentation method
        aug_id = random.randint(0, len(agumenters) - 1)
        augmenter = agumenters[aug_id]

        tasks.append((augmenter, headline, label))
    return tasks


def __augment_headline(args):
    augmenter, original_sentence, label = args
    try:
        augmented_sentence = augmenter.augment(original_sentence)
        return (augmented_sentence[0], label)
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    if not os.path.exists('data/fixed-data'):
        os.makedirs('data/fixed-data')
    fix_sentiment_data()
    fix_topic_data()
