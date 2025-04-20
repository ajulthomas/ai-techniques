import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


MODEL_CONFIG = {
    "nb": {
        "name": "Naive Bayes",
        "model": MultinomialNB(),
    },
    "lr": {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000),
    },
    "rf": {
        "name": "Random Forest",
        "model": RandomForestClassifier(n_estimators=500, random_state=42),
    },
}

CLASS_NAMES = ["FAKE", "TRUE"]


def encode_labels(y):
    """
    Encode the labels to 0 and 1.
    """
    return np.where(y == "FAKE", 0, 1)


def show_results(config=MODEL_CONFIG):
    """
    Show the results of the models.
    """
    for model_key, model_config in config.items():
        result = model_config["result"]
        print(f"Model: {model_config['name']}")
        print(f"Train Accuracy: {result['accuracy_train']:.4f}")
        print(f"Test Accuracy: {result['accuracy_test']:.4f}")
        print("Classification Report:")
        print(result["report"])
        print("Confusion Matrix:")
        cm = result["confusion_matrix"]
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_config['name']}")
        plt.xticks(ticks=[0.5, 1.5], labels=CLASS_NAMES)
        plt.show()


def vectorize_data(X_train, X_test):
    """
    Vectorize the data using TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # save the vectorizer for later use
    # with open("vectorizer.pkl", "wb") as f:
    #     pickle.dump(vectorizer, f)

    return X_train_vectorized, X_test_vectorized, vectorizer


def train_test_models(X_train, y_train, X_test, y_test):
    """
    Train the models and evaluate them.
    """
    for model_key, model_config in MODEL_CONFIG.items():
        model = model_config["model"]
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
        cm = confusion_matrix(y_test, y_test_pred)

        result = {
            "name": model_config["name"],
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "report": report,
            "confusion_matrix": cm,
        }
        MODEL_CONFIG[model_key]["result"] = result
        MODEL_CONFIG[model_key]["model"] = model


def save_models(vectorizer, config=MODEL_CONFIG):
    """
    Save the trained models to disk.
    """
    # save the vectorizer
    with open("../output/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    for model_key, model_config in config.items():
        model = model_config["model"]
        with open(f"../output/{model_key}.pkl", "wb") as f:
            pickle.dump(model, f)


# check if the models already saved
# return true, if they are already saved and load them
# return false, if they are not saved
def load_models():
    try:
        with open("../output/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        for model_key in MODEL_CONFIG.keys():
            with open(f"../output/{model_key}.pkl", "rb") as f:
                model = pickle.load(f)
                MODEL_CONFIG[model_key]["model"] = model
        return True, vectorizer
    except FileNotFoundError:
        return False, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return False, None


# function to test the models
def test_models(X_train, X_test, y_train, y_test):
    """
    Test the models and show the results.
    """
    for model_key, model_config in MODEL_CONFIG.items():
        model = model_config["model"]
        y_train_pred = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        y_test_pred = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
        cm = confusion_matrix(y_test, y_test_pred)

        result = {
            "name": model_config["name"],
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "report": report,
            "confusion_matrix": cm,
        }
        MODEL_CONFIG[model_key]["result"] = result


def train_classic_models(df):
    X = df["cleaned_news"]
    y = encode_labels(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_data(X_train, X_test)
    # check if the models are already saved
    models_loaded, vectorizer = load_models()
    if models_loaded:
        print("Models already trained and saved. Loading them...")
        # test the models
        test_models(X_train_vectorized, X_test_vectorized, y_train, y_test)
    else:
        print("Training models...")
        # train and test the models
        train_test_models(X_train_vectorized, y_train, X_test_vectorized, y_test)
        # save the models
        print("Saving models...")
        save_models(vectorizer)

    show_results()
    return MODEL_CONFIG, vectorizer
