from umap import UMAP
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import ijson
import csv
import random
import sys

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

user_input = sys.argv[1:]
i = int(user_input[0])
seed_words = user_input[1:11]

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
ctfidf_model = ClassTfidfTransformer(
    seed_words=seed_words,
    seed_multiplier=2
)

topic_model = BERTopic(
    umap_model=umap_model,
    min_topic_size=15,
    ctfidf_model=ctfidf_model,
    nr_topics=15,
    top_n_words=30
).fit(docs)

topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])

if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv(f'./modified/seeds_content_topic_info_{i}.tsv', sep='\t', index=False)
document_info.to_csv(f'./modified/seeds_content_document_info_{i}.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")

