import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Saukhyad Mohole\OneDrive - vit.ac.in\Desktop\DBMS RP\1429_1.csv", low_memory=False)

# Drop rows with missing product name or review text
df_clean = df.dropna(subset=['name', 'reviews.text'])

# Remove duplicate reviews
df_clean = df_clean.drop_duplicates(subset=['reviews.text'])

# Remove reviews that are very short (less than 30 characters) - likely not useful
df_clean = df_clean[df_clean['reviews.text'].str.len() > 30]

# Remove leading/trailing whitespace and convert to lowercase for consistency
df_clean['name'] = df_clean['name'].str.strip().str.lower()
df_clean['reviews.text'] = df_clean['reviews.text'].str.strip().str.lower()

# Remove non-alphanumeric characters (optional)
import re
df_clean['reviews.text'] = df_clean['reviews.text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Reset index for clean DataFrame
df_clean = df_clean.reset_index(drop=True)

# Preview cleaned data
print(df_clean[['name', 'reviews.text']].head())

df_clean['content'] = df_clean['name'] + ". " + df_clean['reviews.text']

df_sample = df_clean.head(100)
df_sample['content'] = df_sample['name'] + ". " + df_sample['reviews.text']

