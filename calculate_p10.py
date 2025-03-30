import pandas as pd


def calculate_p10(results_file):

    columns = ["query_id", "Q0", "doc_id", "rank", "score", "tag"]
    df = pd.read_csv(results_file, sep="\t", names=columns, header=0)

    
    # Ensuring rank column is numeric
    df["rank"] = pd.to_numeric(df["rank"], errors='coerce', downcast='integer')  # Safe conversion
   
    
    # dic that stores P@10  scores
    p10_scores = {}

    # decided on using a rleevance threshold to determine what docs are relevant
    #(since determining relevancy can be subjective so can change depending on users opinion)
    relevance_threshold = 0.75

    for query, group in df.groupby("query_id"):
        top_10_docs = group.nsmallest(10, "rank")
        relevant_count = sum(1 for _, row in top_10_docs.iterrows() if row['score'] > relevance_threshold)
        p10_scores[query] = relevant_count / 10  # Computes P@10

    return p10_scores

if __name__ == "__main__":
    
    p10_scores1 = calculate_p10("res/a2_res_USE.txt")
    p10_scores2 = calculate_p10("res/a2_res_BERT.txt")

    if p10_scores1 is not None:
        print("Calculating P10 on file: res/a2_res_USE.txt")

        # Print results
        for query, score in p10_scores1.items():
            print(f"Query {query}: P@10 = {score:.2f}")
    else:
        print("an erroro occured: P@10 score calculation returned None")

    if p10_scores2 is not None:
        print("Calculating P10 on file: res/a2_res_BERT.txt")
        # Print results
        for query, score in p10_scores2.items():
            print(f"Query {query}: P@10 = {score:.2f}")
    else:
        print("an erroro occured: P@10 score calculation returned None")
