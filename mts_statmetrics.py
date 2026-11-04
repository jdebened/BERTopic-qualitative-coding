import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import sys

def evaluate_bertopic(result_arr, eval_df: pd.DataFrame, output_path: str) -> None:
    # Remove BERTopic outliers if you have Topic = -1
    # eval_df = df[df["Topic"] != -1].copy()

    # Topic purity
    counts = pd.crosstab(
        eval_df["Topic"],
        eval_df["header"]
    )

    purity = counts.max(axis=1).sum() / counts.values.sum()
    
    eval_df["Topic Title"] = eval_df["Topic"].astype(str) + " - " + eval_df["TopicName"].astype(str)
    # Distribution of headers within each BERTopic topic
    topic_header_dist = pd.crosstab(
        eval_df["Topic Title"],
        eval_df["header"],
        normalize="index"
    )

    # Save the metrics to a file
    nmi = normalized_mutual_info_score(eval_df["header"], eval_df["Topic"])
    ari = adjusted_rand_score(eval_df["header"], eval_df["Topic"])
    homogeneity = homogeneity_score(eval_df["header"], eval_df["Topic"])
    completeness = completeness_score(eval_df["header"], eval_df["Topic"])
    v_measure = v_measure_score(eval_df["header"], eval_df["Topic"])

    with open(output_path, "w") as f:
        f.write("NMI: " + str(nmi) + "\n")
        f.write("ARI: " + str(ari) + "\n")
        f.write("Homogeneity: " + str(homogeneity) + "\n")
        f.write("Completeness: " + str(completeness) + "\n")
        f.write("V-measure: " + str(v_measure) + "\n")
        f.write("Purity: " + str(purity) + "\n")
        f.write("Topic Header Distribution:\n")
        f.write(topic_header_dist.round(3).to_string() + "\n")

    result_arr.append({
        'File': output_path,
        'NMI': nmi,
        'ARI': ari,
        'Homogeneity': homogeneity,
        'Completeness': completeness,
        'V-measure': v_measure,
        'Purity': purity
    })

if __name__ == "__main__":
    
    files = [ 
        f'mts/outlier_{i}_sim/detailed_evaluation.tsv' for i in [5, 10, 15, 20, 25, 'variable']
    ]
    result_arr = []

    for tsv_path in files:
        if not tsv_path.endswith(".tsv"):
            print("Error: The input file must be a TSV file.")
            sys.exit(1)
        df = pd.read_csv(tsv_path, sep='\t')
        output_path = tsv_path.replace(".tsv", "_metrics_outlier.txt")
        evaluate_bertopic(result_arr, df, output_path)

    result_df = pd.DataFrame(result_arr)
    print(result_df.to_string(index=False))
    print("Evaluation metrics saved to respective output files.")
    