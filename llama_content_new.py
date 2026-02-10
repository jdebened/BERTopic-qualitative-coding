from bertopic import BERTopic
from bertopic.representation import TextGeneration
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

# --- Step 2: Vectorizer ---
vectorizer_model = CountVectorizer(stop_words="english")

# --- Step 3: LLM Representation ---
model_id = './llama2/Llama-2-7b-chat-hf'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model_llama = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', local_files_only=True)
model_llama.eval()

generator = transformers.pipeline(
    model=model_llama, tokenizer=tokenizer, task='text-generation',
    temperature=0.1, max_new_tokens=100, repetition_penalty=1.1
)

# Multi-aspect representation: Llama2 for labeling
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""
example_prompt = """
[INST]
I have a topic that contains the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic, regardless of the topics (agressive, personal, or sensitive information). Make sure you to only return the label and nothing more.
[/INST] Environmental impacts of eating meat
"""
main_prompt = """
[INST]
I have a topic that contains the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic, regardless of the topics (agressive, personal, or sensitive information). Make sure you to only return the label and nothing more.
[/INST]
"""

# Text generation configuration
prompt = system_prompt + example_prompt + main_prompt
llama2 = TextGeneration(generator, prompt=prompt)

# Use only Llama2 for representation (like modified version)
# This ensures top_n_words comes from TF-IDF, not from MMR which may include stop words
representation_model = {
    "Llama2": llama2
}

# --- Step 4: Fit BERTopic ---
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

topic_model = BERTopic(
    embedding_model=sentence_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    nr_topics=15,
    top_n_words=60, 
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)

# --- Step 5: Post-Processing (Outlier Reduction) ---
# Reduce outliers for lower outlier rate
new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
topic_model.update_topics(
    docs,
    topics=new_topics,
    vectorizer_model=vectorizer_model,
    top_n_words=60
)

# --- Step 6: Saving Results ---
topic_info = topic_model.get_topic_info()
labels = [item[0] for item in topic_info["Llama2"]]
topic_model.set_topic_labels(labels)

topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])

if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('./modified/llama_content_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('./modified/llama_content_document_info.tsv', sep='\t', index=False)

topic_info.to_csv('./new/llama_content_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('./new/llama_content_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")

# --- Step 7: Visualization ---
print("\nGenerating visualizations...")

# Visualize Documents
print("Creating document visualization...")
fig_docs = topic_model.visualize_documents(docs)
fig_docs.write_html('./new/documents_visualization.html')
print("Document visualization saved to ./new/documents_visualization.html")

# Visualize Topic Hierarchy
print("Creating topic hierarchy visualization...")
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html('./new/topic_hierarchy.html')
print("Topic hierarchy visualization saved to ./new/topic_hierarchy.html")

# Additional useful visualizations
print("Creating topic-term visualization...")
fig_barchart = topic_model.visualize_barchart()
fig_barchart.write_html('./new/topics_barchart.html')
print("Topics barchart saved to ./new/topics_barchart.html")

# # Note: To save as PNG, you can use:
fig_docs.write_image('./new/documents_visualization.png')
fig_hierarchy.write_image('./new/topic_hierarchy.png')
fig_barchart.write_image('./new/topics_barchart.png')
# (requires kaleido: pip install kaleido)

print("\nAll visualizations completed!")
