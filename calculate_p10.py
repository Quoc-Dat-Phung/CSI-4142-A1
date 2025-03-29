import pandas as pd


def calculate_p10():

    while True:
        try:
            file_Number = input("Pick what file to test P10 on (1 for USE or 2 for BERT): ")
            
            if int(file_Number) == 1:
                results_file = "C:/Users/zakar/OneDrive/Desktop/CSI-4142-A1/res/a2_res_USE.txt"
                break
            elif int(file_Number) == 2:
                results_file = "C:/Users/zakar/OneDrive/Desktop/CSI-4142-A1/res/a2_res_BERT.txt"
                break
            else:
                raise ValueError("Invalid input. Please enter 1 for USE or 2 for BERT.")

        except ValueError as e:
            print(f"Error: {e}. Please try again.")

    print("Calculating P10 on file:"+results_file)
    columns = ["query_id", "Q0", "doc_id", "rank", "score", "tag"]
    df = pd.read_csv(results_file, sep="\t", names=columns, header=0)

    
    # Ensuring rank column is numeric
    df["rank"] = pd.to_numeric(df["rank"], errors='coerce', downcast='integer')  # Safe conversion
   
    
    # dic that stores P@10  scores
    p10_scores = {}

    # decided on using a rleevance threshold to determine what docs are relevant
    #(since determining relevancy can be subjective)
    relevance_threshold = 0.40

    for query, group in df.groupby("query_id"):
        top_10_docs = group.nsmallest(10, "rank")
        relevant_count = sum(1 for _, row in top_10_docs.iterrows() if row['score'] > relevance_threshold)
        p10_scores[query] = relevant_count / 10  # Computes P@10

    return p10_scores

if __name__ == "__main__":
    p10_scores = calculate_p10()

    if p10_scores is not None:  
        # Print results
        for query, score in p10_scores.items():
            print(f"Query {query}: P@10 = {score:.2f}")
    else:
        print("an erroro occured: P@10 score calculation returned None")
