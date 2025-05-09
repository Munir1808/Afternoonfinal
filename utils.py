# -*- coding: utf-8 -*-
"""Untitled17.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kPbRUcGT3eGbETDNM_akeSSPUBMlNjs8
"""

import re
from typing import Tuple, List, Dict
import spacy

# Load spaCy models (make sure you’ve done `pip install spacy` and downloaded these)
nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    # 1) Structured PII via regex
    patterns = {
        "email":            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}',
        "phone_number":     r'\+?[0-9\-]{10,20}',
        "dob":              r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b',
        "aadhar_num":       r'\b\d{4}\s\d{4}\s\d{4}\b',
        "credit_debit_no":  r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
        "expiry_no":        r'\b(0[1-9]|1[0-2])\/\d{2,4}\b',
        "cvv_no":           r'CVV[:\s]*\d{3,4}',
    }
    replacements = {
        "email":           "[email]",
        "phone_number":    "[phone_number]",
        "dob":             "[dob]",
        "aadhar_num":      "[aadhar_num]",
        "credit_debit_no": "[credit_debit_no]",
        "expiry_no":       "[expiry_no]",
        "cvv_no":          "CVV:[cvv_no]",
        "full_name":       "[full_name]",
    }
    # lower number => higher priority
    priority = {
        "full_name":       1,
        "dob":             2,
        "email":           3,
        "credit_debit_no": 4,
        "aadhar_num":      5,
        "cvv_no":          6,
        "expiry_no":       7,
        "phone_number":    8,
    }

    spans = []
    entities = []

    # --- collect regex matches ---
    for label, pat in patterns.items():
        for m in re.finditer(pat, text):
            s, e = m.span()
            val = m.group()
            if label == "cvv_no":
                digits = re.search(r'\d{3,4}', val).group()
                spans.append((s, e, replacements[label], digits, priority[label]))
                entities.append({ "position":[s,e], "classification":label, "entity":digits })
            else:
                spans.append((s, e, replacements[label], val, priority[label]))
                entities.append({ "position":[s,e], "classification":label, "entity":val })

    # --- spaCy PERSON but only right after an intro phrase ---
    intros = ["my name is ", "i am ", "this is "]
    for doc in (nlp_en(text), nlp_de(text)):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                s, e = ent.start_char, ent.end_char
                # check if one of the intros appears *immediately* before
                for intro in intros:
                    L = len(intro)
                    if s >= L and text[s-L:s].lower() == intro:
                        name = ent.text
                        spans.append((s, e, replacements["full_name"], name, priority["full_name"]))
                        entities.append({ "position":[s,e], "classification":"full_name", "entity":name })
                        break

    # --- resolve overlaps by (start, priority) ---
    spans.sort(key=lambda x: (x[0], x[4]))
    final_spans = []
    occupied = set()
    for s, e, rep, val, pr in spans:
        if any(i in occupied for i in range(s, e)):
            continue
        final_spans.append((s, e, rep))
        occupied.update(range(s, e))

    # --- rebuild masked text ---
    result = []
    last = 0
    for s, e, rep in sorted(final_spans, key=lambda x: x[0]):
        result.append(text[last:s])
        result.append(rep)
        last = e
    result.append(text[last:])

    return "".join(result), entities