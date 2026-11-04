import pandas as pd
import numpy as np
import ast
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def evaluate_topic_models(file_list):
    all_results = []

    for filename in file_list:
        try:
            df = pd.read_table(filename)
            # Standardize column cleaning
            df['Representation'] = df['Representation'].apply(ast.literal_eval)
            
            # Split outliers from valid topics
            outliers = df[df['Topic'] == -1]
            valid_topics = df[df['Topic'] != -1]
            topic_words = valid_topics['Representation'].tolist()
            
            # --- 1. Statistical Coherence (Proxy for Topic "Tightness") ---
            dictionary = Dictionary(topic_words)
            corpus = [dictionary.doc2bow(text) for text in topic_words]
            coherence_model = CoherenceModel(
                topics=topic_words, 
                dictionary=dictionary, 
                corpus=corpus, 
                coherence='u_mass'
            )
            coherence = coherence_model.get_coherence()

            # --- 2. Topic Diversity & Anchor Strength ---
            # Diversity: Total unique words / Total words
            all_words = [word for sublist in topic_words for word in sublist]
            diversity = len(set(all_words)) / len(all_words) if all_words else 0
            
            # Anchor Strength: How many topics have unique top-3 keywords?
            # High score means the Distance Map would show distinct, non-overlapping concepts.
            top_3_sets = [tuple(sorted(words[:3])) for words in topic_words]
            anchor_strength = len(set(top_3_sets)) / len(top_3_sets) if top_3_sets else 0

            # --- 3. Intertopic Overlap (Proxy for Distance Map/Clustering) ---
            # Calculates average Jaccard similarity between all topic pairs.
            topic_sets = [set(words) for words in topic_words]
            overlaps = []
            for i in range(len(topic_sets)):
                for j in range(i + 1, len(topic_sets)):
                    intersection = len(topic_sets[i] & topic_sets[j])
                    union = len(topic_sets[i] | topic_sets[j])
                    overlaps.append(intersection / union)
            avg_overlap = np.mean(overlaps) if overlaps else 0

            # --- 4. Dataset Coverage ---
            total_docs = df['Count'].sum()
            outlier_ratio = (outliers['Count'].sum() / total_docs) if total_docs > 0 else 0

            all_results.append({
                "File": filename,
                "Topic Count": len(valid_topics),
                "Outlier %": f"{outlier_ratio*100:.1f}%",
                "Coherence (U_Mass)": round(coherence, 4),
                "Diversity": round(diversity, 4),
                "Anchor Strength": round(anchor_strength, 4),
                "Avg Overlap": round(avg_overlap, 4)
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create comparison table
    comparison_df = pd.DataFrame(all_results)
    return comparison_df

# To run:
files = ['org/llm-contentnews_topic_info.tsv', 
         'new/llama_content_topic_info.tsv',
         'modified/llama_content_topic_info.tsv',
         'modified/seeds_content_topic_info_0.tsv', 
         'modified/semisup_content_topic_info.tsv']
results = evaluate_topic_models(files)
print(results)