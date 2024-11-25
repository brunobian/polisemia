import gensim
from gensim.models import Word2Vec
import plotly.express as px
import numpy as np

# Sample data
sentences = [
    "programa manazana naranja",
    "verde python computadora",
    "codear estreptococo mesa",
    "tabla silla cancer",
    "botella agua marca"
]

# Tokenize sentences
tokenized_sentences = [sentence.split() for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=3, min_count=1, workers=4)

# Retrieve embeddings and corresponding words
words = list(model.wv.index_to_key)
vectors = [model.wv[word] for word in words]

# Reduce dimensions using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
vectors_reduced = pca.fit_transform(vectors)

# Visualize in 3D using Plotly
fig = px.scatter_3d(
    x=vectors_reduced[:, 0], 
    y=vectors_reduced[:, 1], 
    z=vectors_reduced[:, 2],
    text=words,
    title="Word Embeddings Visualized in 3D"
)
fig.update_traces(textposition='top center')
fig.show()
