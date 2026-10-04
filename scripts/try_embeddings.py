from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "Revenue grew by 12% this year.",
    "Sales increased by twelve percent.",
    "The board met four times during the year.",
]

vectors = model.encode(sentences, normalize_embeddings=True)
print("Shape:", vectors.shape)
print("First 5 numbers of sentence 1:", vectors[0][:5])
print()

similarity = vectors @ vectors.T
for i in range(3):
    for j in range(i + 1, 3):
        print(f"{similarity[i][j]:.2f}   {sentences[i]!r}  vs  {sentences[j]!r}")