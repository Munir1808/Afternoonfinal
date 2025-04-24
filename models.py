


pip install pandas numpy scikit-learn joblib tqdm transformers datasets torch






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# ===========================
# Load Data
# ===========================

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Class distribution:\n", df["type"].value_counts())
    return df

# ===========================
# Evaluate ML Model
# ===========================

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, y_pred))

    # Optional Confusion Matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical')
    cm_display.ax_.set_title(f"{name} Confusion Matrix")
    plt.show()

    return acc

# ===========================
# Train ML Models
# ===========================

def train_ml_models(df):
    X = df["email"]
    y = df["type"]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    }

    best_acc = 0
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        acc = evaluate_model(model, X_test, y_test, name)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name

    print(f"\nBest ML model: {best_model_name} with accuracy {round(best_acc * 100, 2)}%")

    joblib.dump(best_model, "ml_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return best_model_name, best_acc

# ===========================
# Transformer Models (BERT / RoBERTa)
# ===========================

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def encode_data(data, tokenizer, max_len=256):
    return tokenizer(data['email'], padding="max_length", truncation=True, max_length=max_len)

def train_transformer_model(df, model_name="bert-base-uncased"):
    print(f"\nTraining Transformer model: {model_name}")

    label2id = {label: i for i, label in enumerate(sorted(df['type'].unique()))}
    id2label = {i: label for label, i in label2id.items()}
    df['label'] = df['type'].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = Dataset.from_pandas(train_df[['email', 'label']])
    test_ds = Dataset.from_pandas(test_df[['email', 'label']])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = train_ds.map(lambda x: encode_data(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: encode_data(x, tokenizer), batched=True)

    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"./{model_name}_model",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        logging_dir=f"./{model_name}_logs",
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    print(f"Starting training for {model_name}...")
    trainer.train()
    print(f"Training completed for {model_name}!")

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=1)
    acc = accuracy_score(test_df['label'], preds)

    print(f"{model_name} Accuracy: {round(acc * 100, 2)}%")

    # Save model & tokenizer
    model.save_pretrained(f"{model_name}_saved")
    tokenizer.save_pretrained(f"{model_name}_saved")

    return model_name, acc

