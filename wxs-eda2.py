# 1. Load Data
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import os

ds = load_dataset("Tobi-Bueck/customer-support-tickets")
df = ds['train'].to_pandas()

os.makedirs("figure2", exist_ok=True)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# 2. Missing Values
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_percent
}).sort_values(by='Missing %', ascending=False)

print("\nMissing Values:\n", missing_df)


# 3. Categorical Analysis
categorical_cols = ['type', 'priority', 'queue', 'language']

for col in categorical_cols:
    print(f"\nValue Counts for {col}:\n")
    print(df[col].value_counts())

    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figure2/{col}_distribution.png", dpi=300)
    plt.close()


# 4. Tag Analysis
from collections import Counter

tags = []
for col in [f'tag_{i}' for i in range(1, 9)]:
    tags.extend(df[col].dropna().tolist())

tag_counts = Counter(tags)

top_tags = tag_counts.most_common(20)

print("\nTop 20 Tags:")
for tag, count in top_tags:
    print(tag, count)

plt.figure()
tags_plot, counts_plot = zip(*top_tags)
plt.barh(tags_plot[::-1], counts_plot[::-1])
plt.title("Top Tags")
plt.tight_layout()
plt.savefig("figure2/top_tags.png", dpi=300)
plt.close()


# 5. Text Length Analysis
df['text_length'] = df['body'].astype(str).apply(len)

plt.figure()
df['text_length'].hist(bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Length")
plt.tight_layout()
plt.savefig("figure2/text_length.png", dpi=300)
plt.close()

print("\nText length stats:")
print(df['text_length'].describe())


# 6. Cross Analysis
print("\nType vs Priority:")
print(pd.crosstab(df['type'], df['priority']))

print("\nLanguage vs Type:")
print(pd.crosstab(df['language'], df['type']))


# 7. NLP-Oriented EDA
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    'please', 'help', 'thanks', 'hello'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['body'].apply(clean_text)


# 8. Word Frequency
all_words = " ".join(df['clean_text']).split()
filtered_words = [w for w in all_words if w not in CUSTOM_STOPWORDS and len(w) > 2]

from collections import Counter
top_words = Counter(filtered_words).most_common(20)

print("\nTop 20 Words:")
for word, count in top_words:
    print(word, count)

plt.figure()
words, counts = zip(*top_words)
plt.barh(words[::-1], counts[::-1])
plt.title("Top Words")
plt.tight_layout()
plt.savefig("figure2/top_words.png", dpi=300)
plt.close()

# 9. Bigram Analysis
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    stop_words='english',
    token_pattern=r'\b[a-z]{3,}\b'
)

X = vectorizer.fit_transform(df['clean_text'])
sum_words = X.sum(axis=0)

bigrams = sorted(
    [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()],
    key=lambda x: x[1],
    reverse=True
)[:20]

print("\nTop 20 Bigrams:")
for bg, count in bigrams:
    print(bg, count)

plt.figure()
bg_words, bg_counts = zip(*bigrams)
plt.barh(bg_words[::-1], bg_counts[::-1])
plt.title("Top Bigrams")
plt.tight_layout()
plt.savefig("figure2/bigrams.png", dpi=300)
plt.close()


# 10. Word Length
df['word_length'] = df['clean_text'].apply(lambda x: len(x.split()))

print("\nWord length stats:")
print(df['word_length'].describe())

plt.figure()
df['word_length'].hist(bins=50)
plt.title("Word Length Distribution")
plt.tight_layout()
plt.savefig("figure2/word_length.png", dpi=300)
plt.close()


# 11. Word Analysis by Type
print("\nTop Words by Ticket Type:")

for t in df['type'].dropna().unique():
    subset = df[df['type'] == t]['clean_text']
    words = " ".join(subset).split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2]

    print(f"\n{t}:")
    print(Counter(words).most_common(10))

