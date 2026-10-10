import pickle


# Load trained model
model_path = "model/mood_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)


def predict_mood(text):
    text = text.strip()

    if not text:
        return None, 0.0

    predicted_mood = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities)) * 100

    return str(predicted_mood), confidence
# -----------------------------
# Test Model
# -----------------------------

if __name__ == "__main__":

    while True:

        user_text = input("\nHow are you feeling? ")

        if user_text.lower() == "exit":
            print("\nExiting Mood Predictor...")
            break

        prediction, results = predict_mood(user_text)

        print("\n==============================")
        print("Predicted Mood :", prediction)
        print("==============================")

        print("\nTop Mood Predictions:")

        for mood, probability in results[:5]:
            print(f"{mood:15} : {probability * 100:.2f}%")