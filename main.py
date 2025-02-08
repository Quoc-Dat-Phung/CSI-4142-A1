# for loading data
import json
import pytrec_eval
import pandas as pd
import copy
import math

# Natural Language Toolkit for text processing
# Source: https://www.nltk.org/
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# for removing common words
nltk.download('stopwords')

# for splitting text into words/sentences
nltk.download('punkt')
nltk.download('punkt_tab')

corpus_file = "scifact/corpus.jsonl"
queries_file = "scifact/queries.jsonl"
test_file = "scifact/qrels/test.tsv"

# columns from each file:
# corpus.jsonl: _id, title, text, metadata
corpus_columns = ["_id", "title", "text", "metadata"]

# queries.jsonl: _id, text, metadata
queries_columns = ["_id", "text", "metadata"]

# test.tsv: query-id, corpus-id, score
test_columns = ["query-id", "corpus-id", "score"]

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def load_stopword_file(filepath):
  words = set()
  with open(filepath, "r", encoding="utf-8") as file:
    for line in file:
      word = line.strip().lower()
      # Ignore empty lines
      if word:  
        words.add(word)
  return stop_words

def is_number(s):
  try:
    # try converting to float
    float(s)  
    return True
  except ValueError:
    return False

def preprocessing(text):
  # 1. Tokenization: Split text into words
  # e.g input "I love reading books in 2025!" -> output ["I", "love", "reading", "books", "in" "2025", "!"]
  tokens = word_tokenize(text)  

  # 2. Lowercase
  # ["I", "love", "reading", "books", "!"] ->  ["i", "love", "reading", "books", "in", "2025", "!"]
  for i in range(len(tokens)): 
    tokens[i] = tokens[i].strip()
    tokens[i] = tokens[i].lower()
  
  # 3. Remove punctuation (and empty strings):
  # e.g ["i", "love", "reading", "books", "!"] -> ["i", "love", "reading", "books", "in", "2025"]
  no_punctuation_tokens = []
  for word in tokens:
    if len(word) > 0 and word.isalpha():
      no_punctuation_tokens.append(word)

  # 4. Remove stopwords
  # ["i", "love", "reading", "books"] -> ["love", "reading", "books", "2025"]
  no_stopwords_tokens = []
  for word in no_punctuation_tokens:
    if word not in stop_words:
      no_stopwords_tokens.append(word)

  # 5. instruction says to remove numbers as well
  # ["love", "reading", "books", "2025"] -> ["love", "reading", "books"]
  text_tokens = []
  for word in no_stopwords_tokens:
    if not is_number(word):
      text_tokens.append(word)

  # 6. Stemming
  # e.g ["love", "reading", "books"] -> ["love", "read", "book"]
  stemmed_tokens = []
  for word in text_tokens:
    stemmed_word = stemmer.stem(word)
    stemmed_tokens.append(stemmed_word)

  return stemmed_tokens

def load_jsonl_file(file_path, columns):
  "load jsonl file then return dataframe"
  data = []
  with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
      data.append(json.loads(line))
  return pd.DataFrame(data, columns=columns)

def load_tsv_file(file_path, columns):
  "Load TSV file then return df"
  return pd.read_csv(file_path, sep="\t", names=columns)

def preprocessing_corpus(df_corpus):
  processed_texts = []

  for i in range(len(df_corpus)):
    # title is important for query searching which is why we combine both the title and the text
    title = df_corpus["title"][i]
    combined_title_text = df_corpus["title"][i] + " " + df_corpus["text"][i]

    # change combined_title_text to title if you only want to include the title in the search
    processed_texts.append(preprocessing(combined_title_text))

  # add this as a new column
  df_corpus["processed_text"] = processed_texts

def build_inverted_index_from_corpus(df_corpus):
  # note that df_corpus must be processed already
  inverted_index = {} 
  
  for i in range(len(df_corpus)):
    doc_id = df_corpus["_id"][i]  
    words = df_corpus["processed_text"][i]  
    
    # loop through words
    for word in words:
        
      # if the word is new, then create an entry doc_id with count 1
      if word not in inverted_index:
        inverted_index[word] = {doc_id: 1}  
          
      # if word isn't new
      else:
        # if the word exists but doc_id is new
        if doc_id not in inverted_index[word]:
          inverted_index[word][doc_id] = 1  
            
        # if both word and doc_id exist, simply increment count
        else:
          inverted_index[word][doc_id] += 1

  
  #Will contain ALL documents in corpus as keys and point to a dictionary of ALL terms in the document with their respective weights 
  #EX: {doc1: {'love':2.131232, 'read':1.35443, 'wolf':4.2745}, doc2:{'love':2.131232, 'hate':7.255}} (almost like the opposite of weighted inverted index) 
  document_vectors={}

  #Copying inverted index to weighted_inverted_index
  weighted_inverted_index = copy.deepcopy(inverted_index)

  #Will contain the highest number/count/frequency for all terms found per document
  max_frequencies = {}

  for _, doc_dict in weighted_inverted_index.items():
    for doc, count in doc_dict.items():
      if doc not in max_frequencies:
        max_frequencies[doc] = count
      else:
        max_frequencies[doc] = max(max_frequencies[doc], count)
  

  
  #N in for formula of log(N/df) (N represents total number of documents in corpus)
  sizeOfCorpus=len(df_corpus)
          
  #this part transforms the inverted_index with raw counts to an inverted index with weights
  for token in weighted_inverted_index:

    #length of the tokens value which is a dictionnary (represents number of documents token is found in)
    numOfDocsTokensAppearIn = len(weighted_inverted_index[token])
    
    if numOfDocsTokensAppearIn == 0:
      continue 
    
    #looping over the documents the token is found in and replacing raw counts to weight
    for doc in weighted_inverted_index[token]:

      tf = weighted_inverted_index[token][doc]/max_frequencies[doc]
      idf = math.log((sizeOfCorpus/numOfDocsTokensAppearIn),2)
      weight=tf*idf
      
      weighted_inverted_index[token][doc]=weight

      if doc not in document_vectors:  
        document_vectors[doc]={token:weight}

      else:
        document_vectors[doc].update({token: weight})

  #returns the weighted inverted index AND the document vectors as a tuple
  return (weighted_inverted_index, document_vectors)

def query_vector_maker_and_retrieval(query, weighted_inverted_index, df_corpus):
    
  #N in for formula of log(N/df) (N represents total number of documents in corpus)
  sizeOfCorpus=len(df_corpus)
  
  #Take the query row and extract the "text" portion of it. We then pass it to the preprocessing function defined above to get the tokens
  query_tokens = preprocessing(query['text'])

  #Constructing the query vector by assigning weights to every Term in the query 
  #tf_query = count_of_term/max frequency in query
  #idf_query= log(size of corpus / number of documents the query's term appears in)
  #weight= tf_query * idf_query
  
  #Find token with highest frequency in query to use in tf formula
  max_query_count = max([query_tokens.count(token) for token in query_tokens])

  #Will be the query vector (query tokens and their weights)
  query_vector={}

  for token in query_tokens:
    tf_query = query_tokens.count(token)/max_query_count

    if token in weighted_inverted_index:
      numOfDocsQueryTokensAppearIn = len(weighted_inverted_index[token])
      idf_query = math.log(sizeOfCorpus / numOfDocsQueryTokensAppearIn, 2)
      query_vector[token] = tf_query * idf_query

    else:
      query_vector[token]=0


  #Creating the retrieved documents dictionnary which will contain ALL documents (that contain 1 or more tokens from the query), the tokens, and the weights  
  #EX/FORMAT:query={A,B,C,D} => { doc1:{A:1.1, B:2.23}, doc2:{B:2.34, C:3.02}, doc3:{A:3.53, B:1.134, C:2.243} }
  retrieved_docs={}
  
  for term in query_tokens:
    if term in weighted_inverted_index:
      for doc_id, weight in weighted_inverted_index[term].items():
        if doc_id not in retrieved_docs:
          retrieved_docs[doc_id] = {}
        
        retrieved_docs[doc_id][term] = weight

  #here for debugging purposes
  # print(query_tokens)
  # pprint.pprint(dict(list(query_vector.items())[:50])) # display 2 items
  # print(len(retrieved_docs))

  #returns the retrieved_docs dictionary AND the query vector as a tuple
  return (retrieved_docs, query_vector)

def ranking(retrieved_docs, query_vector, document_vectors):
    
  #will contain the results of cosine similarity calculations 
  results_dict={}
  
  #will hold length of query and document vectors (euclidean norm)
  query_length=0
  doc_length=0

  #can calculate query vector length here since we only have once query and multiple documents
  query_length = math.sqrt(sum(val**2 for val in query_vector.values()))

  for docs in retrieved_docs:

    #perforing dot product here (Ex: query_vector={x1,x2,x3} and doc_vector= {y1,y2,y3}. query_vector*doc_vector= x1*y1+x2*y2+x3*y3)
    cos_sim_numer = sum(query_vector.get(token, 0) * retrieved_docs[docs].get(token, 0) for token in query_vector)

    #calculating document vector length here
    doc_length = math.sqrt(sum(value**2 for value in document_vectors[docs].values()))
    
    cos_sim_denom = query_length*doc_length

    if cos_sim_denom!=0:
      results_dict[docs] = (cos_sim_numer/cos_sim_denom)
    else:
      results_dict[docs] = 0
      
  #result_dict has to now be ranked in descending order (highest score to lowest) and we need to display top 100 results (I think)
  return results_dict

def run_queries_and_write_to_file(document_vectors, weighted_inverted_index, df_queries, df_corpus):
  res = []

  # We retrieve the documents and the query vector for each query
  for _, row in df_queries.iterrows():
    retrieved_docs, query_vector=query_vector_maker_and_retrieval(row, weighted_inverted_index, df_corpus)
    results=ranking(retrieved_docs, query_vector, document_vectors)
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    res.append([row["_id"], sorted_results])

  # We write the results to a file but for each query only the first 100 results
  with open("results.txt", "w") as file:
      file.write("query_id\tQ0\tdoc_id\trank\tscore\ttag\n")
      for _, r in enumerate(res):
          for j, (doc_id, score) in enumerate(r[1].items()):
              if j >= 100:
                  break
              file.write(f"{r[0]} Q0 {doc_id} {j} {score:.5f} tag_{r[0]}_{doc_id}\n")

def calculate_map():
  # Load qrels/ground truth
  qrels = {}
  with open(test_file, "r") as f:
    next(f)  # Skip header if present
    for line in f:
      qid, docid, rel = line.strip().split()
      if qid not in qrels:
        qrels[qid] = {}
      qrels[qid][docid] = int(rel)

  # Load results 
  run = {}
  with open("results.txt", "r") as f:
    next(f)  # Skip header
    for line in f:
      qid, _, docid, _, score, _ = line.strip().split()
      if qid not in run:
        run[qid] = {}
      run[qid][docid] = float(score)

  # Create evaluator
  evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map'})

  # Calculate scores
  results = evaluator.evaluate(run)

  # Calculate mean scores across all queries
  mean_scores = {}
  for metric in ['map']:
    mean_scores[metric] = sum(query_scores[metric] 
      for query_scores in results.values()) / len(results)

  print(f"Mean Average Precision (MAP): {mean_scores['map']:.4f}")

if __name__ == "__main__":
  stop_words_from_file = load_stopword_file("from_professor/stopwords.txt")
  stop_words.update(stop_words_from_file)

  df_corpus = load_jsonl_file(corpus_file, corpus_columns)
  df_queries = load_jsonl_file(queries_file, queries_columns)
  df_test = load_tsv_file(test_file, test_columns)

  preprocessing_corpus(df_corpus)

  weighted_inverted_index, document_vectors = build_inverted_index_from_corpus(df_corpus)

  run_queries_and_write_to_file(document_vectors, weighted_inverted_index, df_queries, df_corpus)
  calculate_map()