import pandas as pd
import ijson
from sentence_transformers import SentenceTransformer, util
import torch
import sys
import os

# Check command line argument
if len(sys.argv) != 2:
    print("Usage: python cossine.py [folder]")
    sys.exit(1)

folder = sys.argv[1]
output_folder = f'sim_{folder}'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def evaluate_title_meaningfulness(json_path, doc_info_tsv_path):
    # 1. Load the original metadata from JSON
    metadata = []
    with open(json_path, 'r') as f:
        for item in ijson.items(f, 'item'):
            metadata.append({
                "original_title": item.get('title', ''),
                "summary": item.get('summary', '')
            })
    
    # 2. Load your BERTopic document info
    # Ensure this TSV matches the order/length of your JSON
    df_docs = pd.read_table(doc_info_tsv_path)
    
    # Merge metadata onto the topic info
    df_eval = pd.concat([df_docs, pd.DataFrame(metadata)], axis=1)
    
    # 3. Load SBERT Model
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./models")
    
    # 4. Calculate Semantic Similarity
    print("Calculating Alignment Scores...")
    
    # Embeddings for Generated Labels and Ground Truth
    gen_labels = df_eval['CustomName'].tolist()
    orig_titles = df_eval['original_title'].tolist()
    summaries = df_eval['summary'].tolist()
    
    # Encode on GPU with convert_to_tensor=True for speed
    print("Encoding generated labels...")
    emb_gen = model.encode(gen_labels, convert_to_tensor=True)
    print("Encoding original titles...")
    emb_orig = model.encode(orig_titles, convert_to_tensor=True)
    print("Encoding summaries...")
    emb_summ = model.encode(summaries, convert_to_tensor=True)
    
    # Compute pairwise cosine similarities (diagonal only - memory efficient)
    # This computes only row-by-row similarity, avoiding the huge N×N matrix
    print("Computing similarity scores...")
    title_sim = torch.nn.functional.cosine_similarity(emb_gen, emb_orig, dim=1).cpu().tolist()
    summary_sim = torch.nn.functional.cosine_similarity(emb_gen, emb_summ, dim=1).cpu().tolist()
    
    df_eval['Title_Alignment_Score'] = title_sim
    df_eval['Summary_Alignment_Score'] = summary_sim
    
    # 5. Aggregate Results by Topic
    topic_report = df_eval.groupby('Topic').agg({
        'CustomName': 'first',
        'Title_Alignment_Score': 'mean',
        'Summary_Alignment_Score': 'mean',
        'Topic': 'count'
    }).rename(columns={'Topic': 'Doc_Count'})
    
    return topic_report, df_eval

# Run the evaluation
report, detailed_df = evaluate_title_meaningfulness(
    f'news_dialogue.json', 
    f'clean_{folder}/llama_content_document_info.tsv'
)

# Save report to TSV with CustomName renamed to Topic
report_to_save = report.rename(columns={'CustomName': 'Topic'})
report_to_save.to_csv(f'{output_folder}/topic_report.tsv', sep='\t')

# Save detailed evaluation to TSV with CustomName renamed to Topic
detailed_df_to_save = detailed_df.rename(columns={'CustomName': 'Topic'})# Remove unnecessary columns
columns_to_drop = ['Representation', 'Llama2', 'Top_n_words', 'Probability', 'Representative_document']
detailed_df_to_save = detailed_df_to_save.drop(columns=[col for col in columns_to_drop if col in detailed_df_to_save.columns])
detailed_df_to_save.to_csv(f'{output_folder}/detailed_evaluation.tsv', sep='\t')

print(f"\nReport saved to {output_folder}/topic_report.tsv")
print(f"Detailed evaluation saved to {output_folder}/detailed_evaluation.tsv")

print("Report saved to topic_report.tsv")
print("Detailed evaluation saved to detailed_evaluation.tsv")
