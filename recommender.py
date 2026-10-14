import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1. DATASET LOAD ----------
base_path = "archive/BBC News Summary/BBC News Summary/News Articles"

titles, contents, categories = [], [], []

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
            lines = text.strip().split("\n")
            titles.append(lines[0])
            contents.append(text)
            categories.append(category)

df = pd.DataFrame({"title": titles, "content": contents, "category": categories})
print("Total articles:", df.shape[0])

# ---------- 2. TF-IDF + COSINE SIMILARITY ----------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
matrix = tfidf.fit_transform(df["content"])
similarity = cosine_similarity(matrix)

# ---------- 3. RECOMMEND FUNCTION ----------
def recommend(index, n=5):
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return df.iloc[[i for i, _ in scores]][["title", "category"]]

# ---------- 4. ACCURACY CHECK ----------
correct, total = 0, 0
for i in range(0, len(df), 10):
    recs = recommend(i)
    correct += (recs["category"] == df.iloc[i]["category"]).sum()
    total += 5

print("Accuracy:", round(correct / total * 100, 2), "%")

# ---------- 5. USER INPUT (SEARCH BY TITLE) ----------
while True:
    print("\n-----------------------------------")
    keyword = input("Enter a keyword from article title (or 'exit'): ")

    if keyword.lower() == "exit":
        print("Thank you!")
        break

    matches = df[df["title"].str.contains(keyword, case=False, na=False)]

    if matches.empty:
        print("No article found. Try another keyword.")
        continue

    print("\nMatching articles:")
    print(matches[["title", "category"]].head(10))

    idx = int(input("\nEnter the index number of the article you want: "))

    print("\nSelected:", df.iloc[idx]["title"], "|", df.iloc[idx]["category"])
    print("\nRecommended articles:")
    print(recommend(idx))