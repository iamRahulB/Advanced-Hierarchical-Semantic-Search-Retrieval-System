import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer

class ClusteringAndSemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_llm = SentenceTransformer(model_name)
        self.clustering_model = None
        self.sentence_embeddings = None

    def encode_text(self, chunked_text):
        self.sentence_embeddings = self.model_llm.encode(chunked_text)
        return self.sentence_embeddings

    def perform_clustering(self, distance_threshold=1.1):
        self.clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        self.clustering_model.fit(self.sentence_embeddings)
        return self.clustering_model

    def plot_dendrogram(self):
        linkage_matrix = linkage(self.sentence_embeddings, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

    def find_related_sentences(self, query_sentence, sentences):
        query_embedding = self.model_llm.encode([query_sentence])
        similarity_scores = cosine_similarity(query_embedding, self.sentence_embeddings)
        most_similar_index = similarity_scores.argmax()
        query_cluster = self.clustering_model.labels_[most_similar_index]
        related_sentences_indices = [i for i in range(len(sentences)) if self.clustering_model.labels_[i] == query_cluster]
        related_sentences_indices = sorted(related_sentences_indices, key=lambda i: similarity_scores[0][i], reverse=True)
        related_sentences = [sentences[i] for i in related_sentences_indices]
        most_similar_sentence = sentences[most_similar_index]
        return most_similar_sentence, related_sentences

