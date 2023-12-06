import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


df = pd.read_csv("dataset/movies_metadata.csv", usecols=["title", "overview"])[:1000]
# train_model()
model = Doc2Vec.load("models/d2v.model")
document_vectors = [model.infer_vector(word_tokenize(str(doc).lower())) for doc in df["overview"]]
df = pd.DataFrame(cosine_similarity(pd.DataFrame(document_vectors)), index=df["title"], columns=df["title"])
