

from models import load_data, train_ml_models, train_transformer_model

df = load_data("combined_emails_with_natural_pii.csv")

# Train all ML models and compare
train_ml_models(df)

# Train BERT
train_transformer_model(df, model_name="bert-base-uncased")

# Train RoBERTa
train_transformer_model(df, model_name="roberta-base")

