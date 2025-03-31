#!/usr/bin/env python
# coding: utf-8

# In[15]:


#!pip install -U sentence-transformers


# In[16]:


import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np


# In[17]:


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# In[18]:


def load_jsonl_file(file_path, columns):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data, columns=columns)


# In[19]:


corpus_file = "scifact/corpus.jsonl"
queries_file = "scifact/queries.jsonl"

corpus_data = load_jsonl_file(corpus_file, ["_id", "title", "text"])
queries_data = load_jsonl_file(queries_file, ["_id", "text"])


# In[20]:


def get_all_documents(include_title=True):
    if include_title:
        corpus_data["full_text"] = corpus_data["title"] + " " + corpus_data["text"]
    else:
        corpus_data["full_text"] = corpus_data["text"]
    return corpus_data["full_text"].tolist()


# # Define the BERT Retriever Class

# In[21]:


class BERTRetriever:
    def __init__(self):
        # Load the pre-trained BERT model for sentence embeddings
        print("Loading BERT model...")
        self.model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        print("Model loaded.")

    def encode(self, texts, batch_size=64):
        """
        Convert a list of texts (sentences or documents) into BERT embeddings.
        Returns a NumPy array of shape (num_texts, embedding_size).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings

    def index_documents(self, documents):
        """
        Store the documents and compute their embeddings using BERT.
        """
        self.documents = documents                      # Store original texts
        self.document_ids = list(range(len(documents))) # Assign each a numeric ID
        self.document_embeddings = self.encode(documents) # Get embeddings
        return self.document_embeddings

    def search(self, query, top_k=100):
        """
        Search for the top_k most similar documents to the given query.
        Returns a list of (document_id, similarity_score) tuples.
        """
        # Convert query into embedding
        query_embedding = self.encode([query])[0]

        # Compute cosine similarity between query and all documents
        cosine_scores = util.cos_sim(query_embedding, self.document_embeddings)[0]
        cosine_scores = cosine_scores.cpu().numpy()

        # Create a list of (document_id, score) pairs
        results = []
        for i in range(len(cosine_scores)):
            results.append((self.document_ids[i], float(cosine_scores[i])))

        # Sort the results by score in descending order
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if results[j][1] > results[i][1]:
                    results[i], results[j] = results[j], results[i]

        # Return top_k results
        return results[:top_k]

    # We can use this if we want to rerank based on the initial results from assignment 1
    def rerank(self, query, initial_results, documents):
        """Rerank initial results using BERT embeddings."""
        doc_ids = [doc_id for doc_id, _ in initial_results]
        docs_to_rerank = [documents[doc_id] for doc_id in doc_ids]
        
        query_embedding = self.encode([query])[0]
        doc_embeddings = self.encode(docs_to_rerank)
        
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        reranked = []
        for i in range(len(similarities)):
            reranked.append((doc_ids[i], similarities[i]))
    
        reranked.sort(key=lambda x: x[1], reverse=True)
    
        return reranked


# # Format and Save Results

# In[22]:


def format_results(results, document_ids, documents):
    """
    Convert search results into a readable format with doc_id, score, and text.
    """
    formatted_results = []

    for result in results:
        doc_index = result[0]
        score = result[1]
        text = documents[doc_index]
        doc_id = document_ids[doc_index]

        formatted_results.append({
            "doc_id": doc_id,
            "score": f"{score:.4f}",
            "text": text
        })

    return formatted_results


# In[23]:


def main():
    # Load and prepare data
    documents = get_all_documents()
    document_ids = []

    for doc_id in corpus_data["_id"].tolist():
        document_ids.append(int(doc_id))

    queries = queries_data["text"].tolist()
    query_ids = []

    for query_id in queries_data["_id"].tolist():
        query_ids.append(int(query_id))

    # Initialize and index documents using BERT
    retriever = BERTRetriever()
    retriever.index_documents(documents)

    # Open result file for writing
    with open("res/a2_res_BERT.txt", "w") as file:
        file.write("query_id\tQ0\tdoc_id\trank\tscore\ttag\n")

    # Process each query and retrieve top documents
    for i in range(len(queries)):
        query = queries[i]
        query_id = query_ids[i]

        results = retriever.search(query)
        formatted_results = format_results(results, document_ids, documents)

        # Append each result to the output file
        with open("res/a2_res_BERT.txt", "a") as file:
            for rank in range(len(formatted_results)):
                result = formatted_results[rank]
                doc_id = result["doc_id"]
                score = result["score"]
                tag = f"tag_{query_id}_{doc_id}"

                file.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{tag}\n")


# In[ ]:


# Note "Error displaying widget" is just a warning for display. It can be ignored
main()


# In[ ]:




