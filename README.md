# DistilNews

DistilNews is the combination of a topic classifier
([`topic-xdistil-uncased`](https://huggingface.co/hakonmh/topic-xdistil-uncased))
and a sentiment classifier
([`sentiment-xdistil-uncased`](https://huggingface.co/hakonmh/sentiment-xdistil-uncased))
for news headlines, tweets, analyst reports, etc.

## Introduction

This repository shows the code used for creating the dataset, and for training
and evaluating the models. And contains a Pytorch state dicts for manually
downloading the models. The models will also be made available on
[HuggingFace](https://huggingface.co), and can be used directly from there.
The code is available for those who want to train their own models, or create
their own dataset using the labelling pipeline used here.

## Models

Both models are based on
[xtremedistil-l12-h384-uncased](https://huggingface.co/microsoft/xtremedistil-l12-h384-uncased),
which has 33 million parameters. It achieved comparable performance to other,
larger models, while having a much smaller size.

The models were trained for 3 epochs, with a batch size of 50, and a learning
rate of 5e-5. The models were trained on a single NVIDIA RTX 3050 GPU, and
took each about 2-3 hours to train.

### Performance

Here are the performance metrics for the models on the test set:

| Model | Test Set Size | Accuracy | F1 Score |
| --- | --- | --- | --- |
| `topic-xdistil-uncased` | 32 799 | 94.44 % | 92.59 % |
| `sentiment-xdistil-uncased` | 17 527 | 94.59 % | 93.44 % |

## Data

DistilNews uses a diverse range of data sources for its training datasets,
including some of the most popular datasets available on Kaggle and GitHub,
such as [the FinancialPhraseBank dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news),
and [the massive stock news analysis database](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests).
The sentiment dataset contains over 300k entries, while the topic
classification dataset has ~600k entries.

The full list of data sources (with some data overlap between the two models):

<details>
  <summary>Topic data sources</summary>

- <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
- <https://www.kaggle.com/datasets/rmisra/news-category-dataset>
- <https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset>
- <https://www.kaggle.com/datasets/lazrus/headlines-5000>
- <https://www.kaggle.com/datasets/adarshsng/title-and-headline-sentiment-prediction>
- <https://www.kaggle.com/datasets/sulphatet/twitter-financial-news>

</details>

<details>
    <summary>Sentiment data sources</summary>

- <https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news>
- <https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests>
- <https://www.kaggle.com/datasets/notlucasp/financial-news-headlines>
- <https://huggingface.co/datasets/chiapudding/kaggle-financial-sentiment>
- <https://www.kaggle.com/datasets/johoetter/labeled-stock-news-headlines>
- <https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news>
- <https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold>

</details>

### Labelling

The main idea here was to label the data using the OpenAI's Chat GPT API.
GPT-4 was considered, but it was found to expensive for this project, so GPT
3.5 was used instead. Unfortunately, the performance of GPT-3.5 alone fell
short of expectations, leading us to augment it with additional classifiers
to enhance the quality of data labeling

- The sentiment data labeled by taking the subset of headlines classified
with the same sentiment by
[GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5)
and [distilRoberta-financial-sentiment](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis).
  - Where Human labels is available, it is used instead.
- The topic data is labeled by using
[GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5),
[tweet-topic-21-multi](https://huggingface.co/cardiffnlp/tweet-topic-21-multi)
and [topic_classification_04](https://huggingface.co/jonaskoenig/topic_classification_04).

*The final labelled dataset is available on request at <haakonholmen@hotmail.com>*

## Usage

The easiest method is to use the models directly from
[HuggingFace](https://huggingface.co) using pipelines:

```python
from transformers import pipeline

topic_classifier = pipeline("sentiment-analysis",
                            model="hakonmh/topic-xdistil-uncased",
                            tokenizer="hakonmh/topic-xdistil-uncased")
sentiment_classifier = pipeline("sentiment-analysis",
                                model="hakonmh/sentiment-xdistil-uncased",
                                tokenizer="hakonmh/sentiment-xdistil-uncased")
SENTENCE = "Global Growth Surges as New Technologies Drive Innovation and Productivity!"
print(topic_classifier(SENTENCE))
print(sentiment_classifier(SENTENCE))
```

```text
[{'label': 'Economics', 'score': 0.9970171451568604}]
[{'label': 'Positive', 'score': 0.9997037053108215}]
```

Or, if you want to load a model directly:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace with "sentiment-xdistil-uncased" to load the sentiment model
topic_model = AutoModelForSequenceClassification.from_pretrained("hakonmh/topic-xdistil-uncased")
topic_tokenizer = AutoTokenizer.from_pretrained("hakonmh/topic-xdistil-uncased")
topic_model.eval()

SENTENCE = "Global Growth Surges as New Technologies Drive Innovation and Productivity!"
inputs = topic_tokenizer(SENTENCE, return_tensors="pt")
logits = topic_model(**inputs).logits
output = logits.argmax(-1).item()
result = topic_model.config.id2label[output]
print(result)
```

```text
Economics
```
