import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, k=5):
        scores, indices = self.index.search(query_vector, k)
        return scores[0], indices[0]

    def save(self, path):
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
