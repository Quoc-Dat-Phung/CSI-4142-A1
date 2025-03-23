import tensorflow_hub as hub
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_jsonl_file(file_path, columns):
  data = []
  with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
      data.append(json.loads(line))
  return pd.DataFrame(data, columns=columns)

def load_tsv_file(file_path, columns):
  "Load TSV file then return df"
  return pd.read_csv(file_path, sep="\t", names=columns)

corpus_file = "scifact/corpus.jsonl"
queries_file = "scifact/queries.jsonl"
corpus_data = load_jsonl_file(corpus_file, ["_id", "title", "text"])
queries_data = load_jsonl_file(queries_file, ["_id", "text"])

def get_all_documents(include_title=True):
  # Combine title and text
  if include_title:
    corpus_data["full_text"] = corpus_data["title"] + " " + corpus_data["text"]
  else:
    corpus_data["full_text"] = corpus_data["text"]
  
  # Return the documents
  return corpus_data["full_text"].tolist()

def format_results(results, document_ids, documents):
  """Format the search results."""
  formatted_results = []
  for doc_id, score in results:
    text = documents[doc_id]
    doc_id = document_ids[doc_id]
    formatted_results.append({"doc_id": doc_id, "score": f"{score:.4f}", "text": text})
  return formatted_results

class USERetriever:
  def __init__(self):
    # Load the Universal Sentence Encoder model
    print("Loading Universal Sentence Encoder model...")
    self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Model loaded!")
      
  def encode(self, texts):
    """Encode a list of texts into embeddings."""
    # Handle batching for large collections if needed
    if len(texts) > 1000:
      # Process in batches to avoid memory issues
      all_embeddings = []
      for i in range(0, len(texts), 1000):
        batch = texts[i:i+1000]
        embeddings = self.model(batch).numpy()
        all_embeddings.append(embeddings)
      return np.vstack(all_embeddings)
    else:
      return self.model(texts).numpy()
  
  def index_documents(self, documents):
    """Index a collection of documents."""
    self.documents = documents
    self.document_ids = list(range(len(documents)))
    self.document_embeddings = self.encode(documents)
    return self.document_embeddings

  def search(self, query, top_k=100):
    """Search for documents similar to the query."""
    # Encode the query
    query_embedding = self.encode([query])[0]
    
    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
    
    # Sort by similarity
    results = [(self.document_ids[i], similarities[i]) 
              for i in range(len(similarities))]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return results[:top_k]
  
  # we can use this if we want to rerank based on the initial results from assignment 1
  def rerank(self, query, initial_results, documents):
    """Rerank initial results using USE embeddings."""
    doc_ids = [doc_id for doc_id, _ in initial_results]
    docs_to_rerank = [documents[doc_id] for doc_id in doc_ids]
    
    query_embedding = self.encode([query])[0]
    doc_embeddings = self.encode(docs_to_rerank)
    
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    reranked = [(doc_ids[i], similarities[i]) for i in range(len(similarities))]
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked

def main():
  # my current implementation does not use rerank method but it is there for reference
  # it instead goes over all the documents for each query and ranks them using USE embeddings

  documents = get_all_documents()
  document_ids = [int(doc_id) for doc_id in corpus_data["_id"].tolist()]
  queries = queries_data["text"].tolist()
  query_ids = [int(doc_id) for doc_id in queries_data["_id"].tolist()]
  
  retriever = USERetriever()
  retriever.index_documents(documents)

  with open("res/a2_res_USE.txt", "w") as file:
    file.write("query_id\tQ0\tdoc_id\trank\tscore\ttag\n")

  for query in queries:
    results = retriever.search(query)
    formatted_results = format_results(results, document_ids, documents)

    with open("res/a2_res_USE.txt", "a") as file:
      for r in formatted_results:
        q_id = query_ids[queries.index(query)]
        d_id = r['doc_id']
        score = r['score']
        rank = formatted_results.index(r)

        file.write(f"{q_id}\tQ0\t{d_id}\t{rank}\t{score}\ttag_{q_id}_{d_id}\n")

main()