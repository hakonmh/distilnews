# FinALBERT

FinALBERT is a powerful tool that combines a topic classifier and a sentiment classifier based on news headlines. This repository provides all the necessary resources for you to get started.

## Data Sources

FinALBERT uses a diverse range of data sources for its training datasets, including some of the most popular datasets available on Kaggle and GitHub, such as [the FinancialPhraseBank dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news), and [the massive stock news analysis database](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests). The number of headlines is about 300k for the sentiment model and 400k for the topic classification model.

The full list of data sources:

### Sentiment

- <https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news>
- <https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests>
- <https://www.kaggle.com/datasets/notlucasp/financial-news-headlines>
- <https://huggingface.co/datasets/chiapudding/kaggle-financial-sentiment>
- <https://www.kaggle.com/datasets/johoetter/labeled-stock-news-headlines>
- <https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news>
- <https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold>
- <https://github.com/duynht/financial-news-dataset>

### Topic Classification

- <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
- <https://www.kaggle.com/datasets/rmisra/news-category-dataset>
- <https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset>
- <https://www.kaggle.com/datasets/lazrus/headlines-5000>
- <https://www.kaggle.com/datasets/adarshsng/title-and-headline-sentiment-prediction>
- <https://www.kaggle.com/datasets/sulphatet/twitter-financial-news>

## Classification

The sentiment data is pre-classified by taking the subset of headlines classified with the same sentiment by [GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5), [FinBERT Tone](https://huggingface.co/yiyanghkust/finbert-tone), and human classifiers (for those datasets which are pre-classified).

The topic data is pre-classified by [GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5).

## Models

Both models are based on [albert-xxlarge v2](https://huggingface.co/albert-xxlarge-v2).

## Usage

Predict sentence:

```python
import finalbert as fb
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sentence = 'Stock market finished side-ways for the day as GDP reports are released'

model = fb.load_model('sentiment', device=device)
probs = model.predict(sentence, out='probs')
print(probs)

>>> tensor([   0.56,    0.35,    0.2])
```

Sentiment prediction pipeline:

```python
import finalbert as fnb
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipeline = fb.pipeline(
    'file_path.csv', headline_col='headlines', target_col=None, device=device
)
model = fb.load_model('sentiment', device=device)

model.predict(pipeline, out='label')
```

*Note:* `target_col` should be strings with labels `positive`, `negative`, or `neutral`.

Topic prediction pipeline:

```python
import finalbert as fb
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipeline = fb.pipeline(
    'file_path.csv', headline_col='headlines', target_col='topic', device=device
)
model = fb.load_model('topic', device=device)

model.predict(pipeline, out='label')
```

*Note:* `target_col` should be strings with labels `economics`, or `other`.
