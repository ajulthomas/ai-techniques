import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data(df: pd.DataFrame) -> pd.DataFrame:
    # df = pd.read_csv(file_path)
    # df.drop(columns=['subject', 'date'], inplace=True)
    df['text_transformer'] = "<title>" + df['title'] + "<content>" + df['text'] + "<end>"
    df['label'] = np.where(df['label'] == "FAKE", 1, 0)
    return df[['text_transformer', 'label']]

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def create_pipeline(model, tokenizer):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=512,
        truncation=True,
        padding=True
    )

def run_predictions(pipeline, texts: list, label2id: dict):
    raw_preds = pipeline(texts)
    predicted_labels = [label2id[p['label']] for p in raw_preds]
    confidences = [p['score'] for p in raw_preds]
    return predicted_labels, confidences

def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

def roberta_pretrained(df):
    model_name = "hamzab/roberta-fake-news-classification"
    label2id = {"FAKE": 1, "TRUE": 0}
    # id2label = {1: "FAKE", 0: "TRUE"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    df_model = load_data(df)
    print(df_model.head(5))
    print("---"*20)
    tokenizer, model = load_model_and_tokenizer(model_name)
    pred_pipeline = create_pipeline(model, tokenizer)
    
    predicted_labels, confidences = run_predictions(pred_pipeline, df_model["text_transformer"].tolist(), label2id)
    
    df_model["predicted_label"] = predicted_labels
    df_model["confidence"] = confidences

    print(df_model.head(5))
    print("---"*20)

    # save the df_model to a csv file
    df_model.to_csv("../output/roberta_predictions.csv", index=False)
    
    evaluate_model(df_model["label"], df_model["predicted_label"])

