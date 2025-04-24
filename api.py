from fastapi import FastAPI
from pydantic import BaseModel
from utils import mask_pii
import joblib

# Load ML model and vectorizer
model = joblib.load("ml_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class EmailRequest(BaseModel):
    email_body: str

@app.post("/classify-email")
def classify_email(request: EmailRequest):
    email_text = request.email_body

    # Step 1: Mask PII
    masked_text, entities = mask_pii(email_text)

    # Step 2: Vectorize and classify
    vectorized = vectorizer.transform([masked_text])
    prediction = model.predict(vectorized)[0]

    return {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_text,
        "category_of_the_email": prediction
    }
