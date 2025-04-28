from flask import Flask, request, jsonify
import joblib
import spacy
import re
from dateparser import parse as parse_date
from dateparser.search import search_dates
from datetime import datetime
import random
import numpy as np
from date_utils import extract_dates

app = Flask(__name__)


# Load trained models
main_classifier = joblib.load("main_intent_classifier.pkl")
greeting_classifier = joblib.load("greeting_sub_classifier.pkl")

# Load models
#model = joblib.load('intent_model.pkl')
#vectorizer = joblib.load('vectorizer.pkl')
#leave_type_model = joblib.load('leave_type_predictor.pkl')

nlp = spacy.load("en_core_web_sm")

# Greeting responses
greeting_responses = {
    "greeting_morning": [
        "Good morning! 🌞 Hope your day starts bright!",
        "Wishing you a refreshing morning! ☕",
        "Morning vibes! Let’s have a great day ahead.",
        "Hey, good morning! Ready to conquer the day?",
        "Rise and shine! 😊"
    ],
    "greeting_afternoon": [
        "Good afternoon! ☀️ Hope your day’s going well.",
        "Hi there, wishing you a productive afternoon!",
        "Hello! Hope you’re having a smooth afternoon.",
        "Hey! Keep shining this afternoon 🌟"
    ],
    "greeting_evening": [
        "Good evening! 🌇 How was your day?",
        "Hi, wishing you a peaceful evening.",
        "Evening vibes! Hope you had a good day.",
        "Hello! Relax and enjoy your evening 😊"
    ],
    "greeting_general": [
        "Hello there! 👋",
        "Hey! Nice to hear from you.",
        "Hi! How can I assist you today?",
        "Hello! What would you like me to do?",
        "Hey buddy! 😊","What's up?" 
    ]
}

# Leave type mapping
LEAVE_TYPES_MAP = {
    "casual": "Casual Leave",
    "cl": "Casual Leave",
    "sick": "Sick Leave",
    "sl": "Sick Leave",
    "earned": "Earned Leave",
    "el": "Earned Leave",
    "maternity": "Maternity Leave",
    "ml": "Maternity Leave",
    "paternity": "Paternity Leave",
    "pl": "Paternity Leave"
}

# Extract leave type from direct mention
def extract_leave_type(text):
    text_lower = text.lower()
    for keyword, full_form in LEAVE_TYPES_MAP.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            return full_form
    return "Casual Leave"  # fallback to CL

# Extract reason from text
def extract_reason(text):
    text = text.strip()
    text_lower = text.lower()

    # 1. Check for explicit reason phrases (with causal triggers)
    causal_pattern = r'(?:because(?: of)?|coz|cuz|cause|due to|as|since|for|so|that(?: is|’s)? why|in order to|to|on account of)\s+(.*?)(?:[.,;!?]|$)'
    match = re.search(causal_pattern, text_lower, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Handle contractions (i'm, im) and shorthand (pls, plz)
    informal_reasons = r'(im\s+not\s+feeling\s+well|not\s+well|feeling\s+sick|ill|unwell|sick|tired|exhausted|headache|stomach\s+ache|fever|feeling\s+unwell|pls\s+apply|please\s+apply)'
    match = re.search(informal_reasons, text_lower, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 3. Fallback: Extract anything before "can you apply leave"
    apply_keywords = ['apply', 'leave', 'request', 'grant']
    apply_index = min([text_lower.find(k) for k in apply_keywords if k in text_lower] + [len(text)])

    possible_reason = text[:apply_index].strip()
    if possible_reason.lower().startswith(('i ', 'have to', 'need to', 'want to', 'feeling', 'went to', 'going to', 'not well', 'im not')):
        return possible_reason

    return None

# Predict leave type from reason using AI
def predict_leave_type_from_reason(reason):
    if reason:
        try:
            return leave_type_model.predict([reason])[0]
        except:
            pass
    return None

# Process full message
def process_message(message):
    threshold = 0.6
    proba = main_classifier.predict_proba([message])[0]
    max_proba = np.max(proba)
    intent = main_classifier.classes_[np.argmax(proba)]
    print(max_proba)
    if max_proba < threshold:
        intent = "unknown"
    # If intent is 'greetings', classify further and add response
    if intent == "greetings":
        sub_intent = greeting_classifier.predict([message])[0]
        response = random.choice(greeting_responses.get(sub_intent, ["Hello! 👋"]))  # Default to a generic hello
        return {"intent": sub_intent, "entities": {}, "response": response}

    elif intent == "apply_leave":
        leave_type = extract_leave_type(message)
        from_date, to_date = extract_dates(message)
        reason = extract_reason(message)

        # Fallback: Predict leave type from reason if not mentioned directly
        if not leave_type and reason:
            leave_type = predict_leave_type_from_reason(reason)
            if leave_type:
                leave_type = LEAVE_TYPES_MAP.get(leave_type.lower(), leave_type)

        return {
            "intent": intent,
            "entities": {
                "leave_type": leave_type,
                "begin_date": from_date,
                "end_date": to_date,
                "purpose": reason,
                "day_type": "Full Day"
            }
        }
    else:
        return {"intent": "unknown", "entities": {}, "response": "Sorry, I didn’t quite understand that."}

# API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "Message is required"}), 400

    result = process_message(message)
    return jsonify(result)

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
