import os
import numpy as np
import pandas as pd
import random
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
                                  'FinBERT Sentiment']].mode(axis=1)[0]
        else:
            df.rename(columns={'GPT Sentiment': 'Sentiment'}, inplace=True)
        df = df[['Headlines', 'Sentiment']]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['Headlines'] = _clean_text(df['Headlines'])
    df = df.drop_duplicates(subset='Headlines')

    df = augment_headlines(df, num_samples=len(df))

    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('data/fixed-data/sentiment-train.csv', index=False)


def fix_sentiment_test_data():
    df = pd.read_csv('data/classified-data/aspect_based_analysis.csv')

    df = df[df['Topic'].isin(['Economy', 'Stock Market', 'Company', 'Economics'])]
    # Select only rows where all 3 sentiment classifiers agree
    mask = ((df['GPT Sentiment'] == df['FinBERT Sentiment']) &
            (df['GPT Sentiment'] == df['Manual Sentiment']))
    df = df[mask]
    df.rename(columns={'GPT Sentiment': 'Sentiment'}, inplace=True)
    df = df[['Headlines', 'Sentiment']]
    df['Headlines'] = _clean_text(df['Headlines'])
    df = df.sample(frac=1)
    df.to_csv('data/fixed-data/sentiment-test.csv', index=False)


def fix_topic_data():
    topic_files = ['News_Category_Dataset_v3.csv', 'labelled_newscatcher_dataset.csv',
                   'ReutersNews106521.csv']
    dfs = []
    for f in topic_files:
        df = pd.read_csv('data/classified-data/' + f)
        if "Manual Topic" not in df.columns:
            df["Manual Topic"] = "BUSINESS"
        mask = (
            df['Topic'].isin(['Economy', 'Stock Market', 'Company', 'Economics']) &
            df['Manual Topic'].isin(['BUSINESS', 'MONEY', 'TECH', 'TECHNOLOGY'])
        )
        df['Topic'] = np.where(mask, 'Economics', 'Other')
        df = df[['Headlines', 'Topic']]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['Headlines'] = _clean_text(df['Headlines'])
    df = df.drop_duplicates(subset='Headlines')
    train, test = _train_test_split(df)
    train.to_csv('data/fixed-data/topic-train.csv', index=False)
    test.to_csv('data/fixed-data/topic-test.csv', index=False)


def _train_test_split(df, train_size=0.975):
    df = df.sample(frac=1)
    train, test = np.split(df, [int(train_size * len(df))])
    return train, test


def _filter_topic(df):
    """Filter out non-economics topics."""
    if 'Manual Topic' in df.columns:
        mask = (
            df['Topic'].isin(['Economy', 'Stock Market', 'Company', 'Economics']) &
            df['Manual Topic'].isin(['BUSINESS', 'MONEY', 'TECH', 'TECHNOLOGY'])
        )
    else:
        mask = df['Topic'].isin(['Economy', 'Stock Market', 'Company', 'Economics'])
    df = df[mask]
    return df


def _clean_text(headlines: pd.Series):
    """
    This function takes a pandas series containing text headlines as input, and performs the
    following cleaning operations:

    1. Converts all text to lowercase.
    2. Replaces all non-alphanumeric characters except for ".", ",", "!", "?", and "-" with a space.
    3. Replaces multiple consecutive spaces with a single space.
    4. Strips any leading or trailing whitespace from each headline.
    5. Removes the string '- analyst blog' from each headline.

    Parameters:
    headlines (pd.Series): A pandas series containing text headlines to be cleaned.

    Returns:
    pd.Series: A pandas series containing the cleaned text headlines.
    """
    headlines = headlines.str.lower()
    headlines = headlines.str.replace(r'[^a-zA-Z0-9.,?!-]', ' ', regex=True)
    headlines = headlines.str.replace(r'\s+', ' ', regex=True)
    headlines = headlines.str.replace('- analyst blog', '')
    headlines = headlines.str.strip()
    return headlines


def augment_headlines(df, num_samples):
    augmented_sentences = []
    augmented_labels = []

    methods = [WordNetAugmenter(), EmbeddingAugmenter(), EasyDataAugmenter()]

    for _ in range(num_samples):
        i = random.randint(0, len(df) - 1)
        original_sentence = df['Headlines'].iloc[i]
        label = df['Sentiment'].iloc[i]

        # randomly select an augmentation method
        method_id = random.randint(0, len(methods) - 1)
        augmenter = methods[method_id]

        augmented_sentence = augmenter.augment(original_sentence)
        augmented_sentences.append(augmented_sentence[0])
        augmented_labels.append(label)

    new_df = pd.DataFrame({'Headlines': augmented_sentences, 'Sentiment': augmented_labels})
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset='Headlines')
    return df


if __name__ == '__main__':
    if not os.path.exists('data/fixed-data'):
        os.makedirs('data/fixed-data')
    fix_topic_data()
    fix_sentiment_train_data()
    fix_sentiment_test_data()
