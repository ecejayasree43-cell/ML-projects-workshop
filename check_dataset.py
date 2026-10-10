
import pandas as pd

# Load dataset
df = pd.read_csv("data/mood_text_dataset (2).csv")

print("Total rows:", len(df))

# Check exact duplicate sentences
df["clean_text"] = (
    df["text"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

duplicate_mask = df["clean_text"].duplicated(keep=False)
duplicates = df[duplicate_mask].sort_values("clean_text")

print("\nExact duplicate sentences:", 
      df["clean_text"].duplicated().sum())

if not duplicates.empty:
    print("\nDuplicate sentence details:")
    print(duplicates[["text", "mood"]].to_string(index=False))
else:
    print("\nNo exact duplicate sentences found.")

# Check same sentence assigned to different moods
mood_counts = df.groupby("clean_text")["mood"].nunique()
conflicts = mood_counts[mood_counts > 1]

print("\nSentences assigned to multiple moods:", len(conflicts))

if not conflicts.empty:
    print("\nConflicting sentences:")
    for sentence in conflicts.index:
        print("\nSentence:", sentence)
        print(df.loc[
            df["clean_text"] == sentence, ["text", "mood"]
        ].to_string(index=False))

print("\nDataset quality check completed!")

# check

# Inspect sentences from weaker moods

weak_moods = [
    "Motivational",
    "Uplifting",
    "Energetic",
    "Romantic",
    "Melancholic"
]

for mood in weak_moods:
    print(f"\n{'=' * 15} {mood} {'=' * 15}")

    samples = df.loc[df["mood"] == mood, "text"].head(10)

    for i, sentence in enumerate(samples, start=1):
        print(f"{i}. {sentence}")
