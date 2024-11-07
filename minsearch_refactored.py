import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

class HybridSearchEngine:
    """
    A hybrid search engine that combines TF-IDF text search with keyword filtering capabilities.
    Uses cosine similarity for text matching and exact matching for keywords.

    Attributes:
        text_fields (list): List of text field names to search.
        keyword_fields (list): List of keyword field names for filtering.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        documents (list): List of indexed documents.
    """

    def __init__(self, text_fields, keyword_fields, vectorizer_params={}):
        """
        Initializes the search engine with specified text and keyword fields.

        Args:
            text_fields (list): List of text field names to search.
            keyword_fields (list): List of keyword field names for filtering.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.documents = []

    def index_documents(self, documents):
        """
        Indexes the provided documents for searching.

        Args:
            documents (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.documents = documents
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in documents]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)

        for doc in documents:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search_documents(self, query, filters={}, field_weights={}, num_results=10):
        """
        Searches the indexed documents with the given query, filters, and field weights.

        Args:
            query (str): The search query string.
            filters (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            field_weights (dict): Dictionary of weight multipliers for text fields. Keys are field names and values are the weights.
            num_results (int): The number of top results to return. Defaults to 10.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        query_vectors = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        relevance_scores = np.zeros(len(self.documents))

        # Compute cosine similarity for each text field and apply weights
        for field, query_vector in query_vectors.items():
            similarity = cosine_similarity(query_vector, self.text_matrices[field]).flatten()
            weight = field_weights.get(field, 1)
            relevance_scores += similarity * weight

        # Apply keyword filters
        for field, value in filters.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                relevance_scores = relevance_scores * mask.to_numpy()

        # Get top results
        top_indices = np.argpartition(relevance_scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-relevance_scores[top_indices])]

        # Filter out zero-score results
        top_documents = [self.documents[i] for i in top_indices if relevance_scores[i] > 0]

        return top_documents
