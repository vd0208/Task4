import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('customer_support_tickets.csv')
import os
os.makedirs("figures", exist_ok=True)

# 2. Basic Overview
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns)

print("\nInfo:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

# 3. Missing Values Analysis
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_percent
}).sort_values(by='Missing %', ascending=False)

print("\nMissing Values:\n", missing_df)

# 4. Numerical Analysis
print("\nNumerical Summary:\n", df.describe())

# Age Distribution
plt.figure()
df['Customer Age'].hist()
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/age_distribution.png", dpi=300)
plt.close()

# Satisfaction Distribution
plt.figure()
df['Customer Satisfaction Rating'].hist()
plt.title("Customer Satisfaction Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/satisfaction_distribution.png", dpi=300)
plt.close()

# 5. Categorical Analysis
categorical_cols = [
    'Ticket Type',
    'Ticket Priority',
    'Ticket Channel',
    'Ticket Status',
    'Product Purchased'
]

for col in categorical_cols:
    print(f"\nValue Counts for {col}:\n")
    print(df[col].value_counts())

    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
    plt.show()

# 6. Bivariate Analysis

# Priority vs Satisfaction
priority_vs_sat = df.groupby('Ticket Priority')['Customer Satisfaction Rating'].mean()
print("\nPriority vs Satisfaction:\n", priority_vs_sat)

plt.figure()
priority_vs_sat.plot(kind='bar')
plt.title("Average Satisfaction by Priority")
plt.ylabel("Satisfaction")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/priority_vs_satisfaction.png", dpi=300)
plt.close()

# Channel vs Ticket Type
channel_type = pd.crosstab(df['Ticket Channel'], df['Ticket Type'])
print("\nChannel vs Ticket Type:\n", channel_type)

# 7. Date Processing
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')

print("\nDate Conversion Done")
print(df['Date of Purchase'].head())

# 8. Text Analysis (Basic)

# Text Length
df['text_length'] = df['Ticket Description'].astype(str).apply(len)

plt.figure()
df['text_length'].hist()
plt.title("Text Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# Word Frequency
from collections import Counter

all_text = " ".join(df['Ticket Description'].astype(str)).lower()
words = all_text.split()

word_counts = Counter(words)
top_words = word_counts.most_common(20)

print("\nTop 20 Words:\n", top_words)

# 9. Save Cleaned Data (Optional)
df.to_csv('cleaned_customer_support_tickets.csv', index=False)

print("\nEDA Completed Successfully!")


# 10. NLP-Oriented EDA

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from collections import Counter

print("\n=== NLP-Oriented EDA ===")

# 10.1 Text Cleaning
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    'productpurchased', 'im', 'ive', 'assist', 'having', 'please'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\{[^}]+\}', '', text)  # remove template placeholders
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Ticket Description'].apply(clean_text)

# 10.2 Improved Word Frequency
all_words = " ".join(df['clean_text']).split()
filtered_words = [w for w in all_words if w not in CUSTOM_STOPWORDS and len(w) > 2]

top_words_clean = Counter(filtered_words).most_common(20)

print("\nTop 20 Cleaned Words:")
for word, count in top_words_clean:
    print(f"{word}: {count}")

plt.figure()
words, counts = zip(*top_words_clean)
plt.barh(words[::-1], counts[::-1])
plt.title("Top Words (Cleaned)")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("figures/top_words.png", dpi=300)
plt.close()

# 10.3 Bigram Analysis
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    stop_words='english',
    token_pattern=r'\b[a-z]{3,}\b'
)

X = vectorizer.fit_transform(df['clean_text'])
sum_words = X.sum(axis=0)

bigram_freq = sorted(
    [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()],
    key=lambda x: x[1], reverse=True
)

print("\nTop 20 Bigrams:")
for bigram, count in bigram_freq[:20]:
    print(f"{bigram}: {count}")

plt.figure()
bg_words, bg_counts = zip(*bigram_freq[:20])
plt.barh(bg_words[::-1], bg_counts[::-1])
plt.title("Top Bigrams")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("figures/bigrams.png", dpi=300)
plt.close()


# 10.4 Text Length
df['word_length'] = df['clean_text'].apply(lambda x: len(x.split()))

print("\nText Length (word-based):")
print(df['word_length'].describe())

plt.figure()
df['word_length'].hist(bins=50)
plt.title("Text Length (Words)")
plt.xlabel("Number of Words")
plt.tight_layout()
plt.savefig("figures/text_length.png", dpi=300)
plt.close()


# 10.5 Word Analysis by Ticket Type
print("\nTop Words by Ticket Type:")

for t in df['Ticket Type'].unique():
    subset = df[df['Ticket Type'] == t]['clean_text']
    words = " ".join(subset).split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2]

    print(f"\n{t}:")
    print(Counter(words).most_common(10))