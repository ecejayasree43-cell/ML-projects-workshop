import pandas as pd
import os
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD MOOD TEXT DATASET
# ==========================================

data = pd.read_csv("data/mood_text_dataset (2).csv")

print("\nDataset Shape:", data.shape)

print("\nMood Distribution:")
print(data["mood"].value_counts())

print("\nMissing Values:")
print(data.isnull().sum())


# ==========================================
# 2. INPUT AND TARGET
# ==========================================

X = data["text"]
y = data["mood"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ==========================================
# 4. TF-IDF + NAIVE BAYES
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 8. SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

model_path = "model/mood_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")

print("Location:", model_path)


# ==========================================
# 9. TEST WITH NEW USER INPUTS
# ==========================================

 
test_sentences = [
    "I had a stressful day and just want something peaceful",
    "I am thinking about the person I love and want a romantic song",
    "I need powerful music for my workout",
    "I want something that motivates me to achieve my goals",
    "I feel sad and lonely tonight",
    "I want a fun and catchy song for a party",
    "I am remembering my childhood and old school days",
    "I want a traditional village style Tamil song",
    "I believe tomorrow will be better and I need some hope",
    "I want a prayer song with a divine feeling"
]

print("\n==========================================")
print("NEW USER INPUT TEST")
print("==========================================")

for sentence in test_sentences:
    prediction = model.predict([sentence])[0]
    probability = max(model.predict_proba([sentence])[0])

    print("\nInput :", sentence)
    print("Mood  :", prediction)
    print(f"Confidence: {probability * 100:.2f}%")


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print("\nMood order:")
print(list(model.classes_))

print("\nConfusion Matrix:")
print(cm)
