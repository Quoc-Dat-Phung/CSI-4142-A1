import pytrec_eval

# change these values accordingly
test_file = "scifact/qrels/test.tsv"
results_file = "C:/Users/zakar/OneDrive/Desktop/CSI-4142-A1/res/a2_res_BERT.txt"
#"C:/Users/zakar/OneDrive/Desktop/CSI-4142-A1/res/a2_res_USE.txt"
def calculate_map(results_file):
  # Load qrels truth
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
  with open(results_file, "r") as f:
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
  print("MAP for USE code:")
  calculate_map("res/a2_res_USE.txt")
  print("MAP for BERT code:")
  calculate_map("res/a2_res_BERT.txt")
