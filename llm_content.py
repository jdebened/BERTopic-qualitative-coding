from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
import ijson
import csv
import random

summaries = []
with open('news_dialogue.json', 'r') as f:
    for index, item in enumerate(ijson.items(f, 'item')):
        string = '\n'.join(item['utt'])
        summaries.append(string)   

categories = []
with open('summary_news_document_info.tsv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)  
    for row in reader:
        topic = int(row[0])  
        categories.append(topic)

with open('summary_news_topic_info.tsv', 'r', encoding='utf-8') as file1:
    num_topic = sum(1 for line in file1) - 1

def get_k_elements(n, k):
    if k > n + 1:
        raise ValueError("k cannot be larger than n + 1")
    return random.sample(range(-1, n + 1), k)

# indices = get_k_elements(num_topic, num_topic // 4)
docs = summaries
# y = [label if label in indices else -1 for label in categories]
            
from sentence_transformers import SentenceTransformer

model_path = './all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_path)

#topic_model = BERTopic(verbose=True, embedding_model=sentence_model).fit(docs, y=y)

# topic_labels = topic_model.generate_topic_labels(nr_words = 3, topic_prefix = False, word_length = 15, separator = " - ")
# topic_model.set_topic_labels(topic_labels)

from sentence_transformers import SentenceTransformer

model_path = './all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_path)
embeddings = sentence_model.encode(docs, show_progress_bar=True)

from transformers import pipeline
from bertopic.representation import TextGeneration
prompt = """
I have a topic described by the following keywords: '[KEYWORDS]'.
Based on the information about the topic above, please create a short label of this topic. Make sure you return only the label. Use consistent structure and noun-based phrasing.
"""
generator = pipeline('text2text-generation', model='google/flan-t5-base', device='cuda')

representation_model = {
    "FlanT5 - Base": TextGeneration(generator, prompt=prompt)
}
topic_model = BERTopic(embedding_model=sentence_model,
                       representation_model=representation_model,
                       # Hyperparameters
                       top_n_words=10,
                       verbose=True)
topics, probs = topic_model.fit_transform(docs, embeddings)
topic_info = topic_model.get_topic_info()
labels = [item[0] for item in topic_info["FlanT5 - Base"]]
topic_model.set_topic_labels(labels)
topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])

if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('./llm_label/llm-contentnews_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('./llm_label/llm-contentnews_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")
