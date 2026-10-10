import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/mood_text_dataset.csv")

print("\nDataset Shape:", data.shape)

X = data["text"]
y = data["mood"]


# ==========================================
# 2. TRAIN / TEST SPLIT
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
# 3. DEFINE MODELS
# ==========================================

models = {

    "Naive Bayes": Pipeline([
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
    ]),

    "Logistic Regression": Pipeline([
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
            LogisticRegression(
                max_iter=2000,
                C=2
            )
        )
    ]),

    "Linear SVM": Pipeline([
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
            LinearSVC(
                C=1.5
            )
        )
    ])
}


# ==========================================
# 4. TRAIN AND COMPARE
# ==========================================

results = {}

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 5. FIND BEST MODEL
# ==========================================

best_model_name = max(
    results,
    key=results.get
)

best_accuracy = results[best_model_name]


print("\n==========================================")
print("FINAL COMPARISON")
print("==========================================")

for name, accuracy in results.items():
    print(f"{name:20} : {accuracy * 100:.2f}%")

print("\n------------------------------------------")
print("BEST MODEL")
print("------------------------------------------")

print("Model   :", best_model_name)
print(f"Accuracy: {best_accuracy * 100:.2f}%")