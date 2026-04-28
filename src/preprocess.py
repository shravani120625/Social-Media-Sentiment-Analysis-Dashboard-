import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)     # remove links
    text = re.sub(r"[^a-z\s]", "", text)    # remove special chars
    text = re.sub(r"\s+", " ", text)        # remove extra spaces
    return text.strip()

# Load dataset
df = pd.read_csv("data/social_media_multi_platform_dataset.csv")

# Clean text column
df["clean_text"] = df["text"].apply(clean_text)

# Standardize sentiment
df["sentiment"] = df["sentiment"].str.lower()

print(df.head())