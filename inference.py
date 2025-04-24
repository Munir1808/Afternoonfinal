#!/usr/bin/env python
# coding: utf-8

# In[3]:


from models import load_data  # Only needed for local testing
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from utils import mask_pii

def classify_email(email_text, model_type="transformer", transformer_name="bert-base-uncased"):
    # Step 1: Mask the input email
    masked_email, entities = mask_pii(email_text)

    # Step 2: Load the model & classify
    if model_type == "ml":
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("ml_model.pkl")
        vec = vectorizer.transform([masked_email])
        label = model.predict(vec)[0]
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{transformer_name}_saved")
        model = AutoModelForSequenceClassification.from_pretrained(f"{transformer_name}_saved")
        model.eval()
        inputs = tokenizer(masked_email, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            label_id = torch.argmax(outputs.logits, dim=1).item()
            label = model.config.id2label[label_id]

    # Step 3: Return response in strict format
    return {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": label
    }

