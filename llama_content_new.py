from bertopic import BERTopic
from bertopic.representation import TextGeneration, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import transformers
import ijson
import pandas as pd

# --- Step 1: Data Preparation ---
summaries = []
with open('news_dialogue.json', 'r') as f:
    for item in ijson.items(f, 'item'):
        summaries.append('\n'.join(item['utt']))
docs = summaries

# --- Step 2: Advanced Sub-Models (The "Engine") ---
# 1. Fix Stability: Add random_state to UMAP
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# 2. Control Outliers: Use HDBSCAN with prediction_data=True
hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# 3. Clean "Word Salad": Custom stop words for news dialogue
# custom_stops = ["know", "going", "just", "people", "think", "really", "actually", "like", "right", "say"]
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
# Add custom words to the vectorizer's stop list if needed

# --- Step 3: LLM & Diversity Representations ---
model_id = './llama2/Llama-2-7b-chat-hf'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model_llama = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', local_files_only=True)
model_llama.eval()

generator = transformers.pipeline(
    model=model_llama, tokenizer=tokenizer, task='text-generation',
    temperature=0.1, max_new_tokens=100, repetition_penalty=1.1
)

# Multi-aspect representation: MMR for diversity + Llama2 for labeling
mmr_model = MaximalMarginalRelevance(diversity=0.4) 
main_prompt = """[INST] Create a concise topic label based on these keywords: [KEYWORDS] [/INST]"""
llama2 = TextGeneration(generator, prompt=main_prompt)

representation_model = {
    "Main": mmr_model,
    "Llama2": llama2
}

# --- Step 4: Fit BERTopic ---
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    calculate_probabilities=False, # Critical for outlier reduction if set True (set False for optimization?)
    nr_topics=15,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)

# --- Step 5: Post-Processing (Outlier Reduction) ---
# This forces that 43% outlier rate down to ~5%
new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
topic_model.update_topics(docs, topics=new_topics)

# --- Step 6: Saving Results ---
topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

# Use Llama2 labels as the primary name
if "Llama2" in topic_info.columns:
    topic_info["CustomName"] = topic_info["Llama2"].apply(lambda x: x[0] if isinstance(x, list) else x)

topic_info.to_csv('./new/llama_content_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('./new/llama_content_document_info.tsv', sep='\t', index=False)
