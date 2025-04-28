
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import joblib

# Your dataset
data = {
    "text": [
        # Leave Intents
        "I want to apply for leave", "Can I take a day off?", "leave", "Need sick leave for tomorrow",
        "Apply leave for me", "Please apply leave", "Apply sick leave", "I need to take leave",
        "Can you apply leave for me on 23rd feb?", "I was sick, need leave", 
        "Apply leave as I was not feeling well", "Apply leave on 23rd Feb due to sickness",
        "Take a leave on my behalf", "I need a day off on 23rd Feb", 
        "I want to request a sick leave", "I want to apply leave because I was ill",
        "Can I take a medical leave?", "I was not well yesterday", 
        "Mark me on leave for yesterday", "Sick leave for yesterday please",
        "Feeling unwell, apply leave", "I had fever, apply leave", "Please record a sick leave",
        "Apply one-day leave", "Leave request for 23 Feb", "Apply leave due to health issue",
        "I wasn’t feeling well yesterday", "Can you apply one day leave?", "Need leave for 23rd Feb",
        "Leave required for 23 Feb due to sickness", "Apply my leave for yesterday","maternity leave","paternal leave"

        # Greetings
        "Hello!", "Hi there", "Hey", "Hey, how are you?", "Good morning", 
        "What's up?", "Hello buddy!", "Greetings", "Hey there!", "Hi",
        "morning", "morning!", "gm", "good morning", "good morning!", "hi!", "hey!", "hello there!",
        "Hi, how’s it going?", "Yo!", "Hello team", "Hey team", "Hi everyone", "Hey buddy", 
        "Hi!", "Greetings of the day", "Hi there!", "Good day!", "Wassup!", "Howdy!",
        "Good evening", "Nice to see you!", "Hello sir", "Hi boss", "Hey chief",

        # Balance
        "What is my balance?", "Show me my account balance", 
        "How much money do I have?", "Can you tell me my balance?", 
        "What's my available leave?", "Tell me my remaining balance", 
        "How many leaves are left?", "Leave balance please",
        "What’s my leave status?", "How many sick leaves do I have left?", 
        "Balance of casual leaves?", "Available leaves?", "Do I have any leave left?",
        "Remaining leave count?", "Show leave details", "What are my current leave days?",
        "Leave quota left?", "What is my casual leave balance?", "Check my earned leave balance",
        "Show leave summary", "Leave report please", "Tell me remaining leave"
    ],
    "intent": [
        *["apply_leave"] * 32,
        *["greetings"] * 35,
        *["fetch_balance"] * 22
    ]
}

# Create a pipeline: TF-IDF + Classifier
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression())
])

# Train the model
pipeline.fit(data["text"], data["intent"])
# Save main classifier
joblib.dump(pipeline, "main_intent_classifier.pkl")


# ======== Step 2: Train Greeting Sub-Classifier ========

# Greeting sub-intents dataset
greeting_data = {
    "text": [
        "Hi", "Hello there", "Hey", "Hello", "Hi there",
        "Good morning", "Morning!", "Morning vibes", "Good morning 🌞", "Wishing you a refreshing morning",
        "Good afternoon", "Afternoon!", "Hope your afternoon is going well", "Have a nice afternoon",
        "Good evening", "Evening!", "Hope you had a great day", "Wishing you a lovely evening"
    ],
    "intent": [
        "greeting_general", "greeting_general", "greeting_general", "greeting_general", "greeting_general",
        "greeting_morning", "greeting_morning", "greeting_morning", "greeting_morning", "greeting_morning",
        "greeting_afternoon", "greeting_afternoon", "greeting_afternoon", "greeting_afternoon",
        "greeting_evening", "greeting_evening", "greeting_evening", "greeting_evening"
    ]
}

# Create a pipeline: TF-IDF + Classifier
greeting_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression())
])

# Train the model
greeting_pipeline.fit(greeting_data["text"], greeting_data["intent"])
# Save main classifier
joblib.dump(greeting_pipeline, "greeting_sub_classifier.pkl")

print("Main Intent Classifier and Greeting Sub-Classifier trained and saved successfully!")
