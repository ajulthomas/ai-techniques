import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.drop(columns=['subject', 'date'], inplace=True)
    df['news'] = "<title>" + df['title'] + "<content>" + df['text'] + "<end>"
    return df[['news', 'target']]

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

def main():
    model_name = "hamzab/roberta-fake-news-classification"
    label2id = {"FAKE": 1, "TRUE": 0}
    
    df_model = load_data("./data/cleaned_fake_real_news.csv")
    tokenizer, model = load_model_and_tokenizer(model_name)
    pred_pipeline = create_pipeline(model, tokenizer)
    
    predicted_labels, confidences = run_predictions(pred_pipeline, df_model["news"].tolist(), label2id)
    
    df_model["predicted_label"] = predicted_labels
    df_model["confidence"] = confidences
    
    evaluate_model(df_model["target"], df_model["predicted_label"])

if __name__ == "__main__":
    main()
