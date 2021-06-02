import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI

def fit_logit():
    df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
    y = df["feeling"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=1
    )

    model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=100)
    model.fit(X_train, y_train)
    return model


def predict_input(input, df):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
    enc = vectorizer.transform([input])
    model = fit_logit()
    pred = model.predict(enc)[0]
    return pred


app = FastAPI()


@app.get("/")
async def root():
    query = "i like"
    response = predict_input(query)
    models = {}
    for el in response:
        models[el[0]] = str(el[1])

    output = {
        "query": query,
        "response": models
    }
    return output


@app.get("/prediction/{query}")
def get_prediction(query):
    response = predict_input(query)
    models = {}
    for el in response:
        models[el[0]] = str(el[1])

    output = {
        "query": query,
        "response": models
    }
    return output
