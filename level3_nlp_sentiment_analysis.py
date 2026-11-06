# %% [markdown]
# # Level 3 - Task 3: NLP Sentiment Analysis  (RUN THIS IN GOOGLE COLAB OR LOCALLY)
# Dataset: NLTK's built-in movie_reviews corpus (2,000 labelled reviews: 1,000 positive / 1,000 negative).
# NOTE: written for an environment with internet access. It could not be executed in the sandbox where the
# other files were built (nltk, TextBlob and wordcloud were unavailable there), so run it once and check the output.
#
# Colab setup:  !pip install nltk textblob wordcloud

# %% Setup
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk, pandas as pd, seaborn as sns
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.metrics import classification_report, accuracy_score

for pkg in ["movie_reviews", "stopwords", "punkt", "punkt_tab", "wordnet"]:
    nltk.download(pkg, quiet=True)

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# %% Load
rows = [(movie_reviews.raw(f), cat) for cat in movie_reviews.categories() for f in movie_reviews.fileids(cat)]
df = pd.DataFrame(rows, columns=["review", "true_label"]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df.shape, df.true_label.value_counts().to_dict())

# %% Preprocess: tokenise, remove stopwords, lemmatise
stop, lem = set(stopwords.words("english")), WordNetLemmatizer()
def clean(text):
    toks = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    return [lem.lemmatize(t) for t in toks if t not in stop and len(t) > 2]
df["tokens"] = df.review.apply(clean)
df["clean_text"] = df.tokens.apply(" ".join)

# %% Sentiment with TextBlob (polarity > 0.05 positive, < -0.05 negative, otherwise neutral)
df["polarity"] = df.review.apply(lambda t: TextBlob(t).sentiment.polarity)
df["sentiment"] = pd.cut(df.polarity, [-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"])
print(df.sentiment.value_counts())

# Sanity check against human labels (neutral rows excluded because the corpus has only pos/neg labels)
sub = df[df.sentiment != "neutral"]
y_pred = sub.sentiment.map({"negative": "neg", "positive": "pos"})
print("Accuracy vs labels:", round(accuracy_score(sub.true_label, y_pred), 3))
print(classification_report(sub.true_label, y_pred))

# %% Visualise
plt.figure(figsize=(6, 4.5))
sns.countplot(data=df, x="sentiment", hue="sentiment", order=["negative", "neutral", "positive"],
              palette=["#e76f51", "#e9c46a", "#2a9d8f"], legend=False)
plt.title("Sentiment Distribution (TextBlob)"); plt.xlabel("Sentiment"); plt.ylabel("Number of reviews")
plt.tight_layout(); plt.savefig(FIG / "sentiment_distribution.png", dpi=250); plt.close()

plt.figure(figsize=(8, 4.5)); sns.histplot(df.polarity, bins=40, kde=True, color="#264653")
plt.title("Polarity Score Distribution"); plt.xlabel("Polarity (-1 negative to +1 positive)")
plt.tight_layout(); plt.savefig(FIG / "polarity_histogram.png", dpi=250); plt.close()

for s, cmap in [("positive", "Greens"), ("negative", "Reds")]:
    text = " ".join(df.loc[df.sentiment == s, "clean_text"])
    wc = WordCloud(width=1000, height=500, background_color="white", colormap=cmap, max_words=120).generate(text)
    plt.figure(figsize=(10, 5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title(f"Most Frequent Words in {s.title()} Reviews", fontsize=14)
    plt.tight_layout(); plt.savefig(FIG / f"wordcloud_{s}.png", dpi=250); plt.close()

df.drop(columns=["tokens"]).to_csv(HERE / "sentiment_results.csv", index=False)
