import pandas as pd


def recommend_songs(mood):

    # Load dataset
    songs = pd.read_csv("data/songs.csv")

    # Remove unwanted spaces and make comparison case-insensitive
    songs["mood"] = songs["mood"].str.strip().str.lower()

    # Find matching songs
    recommended = songs[
        songs["mood"] == mood.strip().lower()
    ]

    return recommended


if __name__ == "__main__":


    result = recommend_songs("Romantic")

    print("\nRecommended Songs:\n")
    print(result)