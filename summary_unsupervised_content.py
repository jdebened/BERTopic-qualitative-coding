from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
import ijson

summaries = []
with open('news_dialogue.json', 'r') as f:
    for index, item in enumerate(ijson.items(f, 'item')):
        # Process each `item` here
        if 'title' in item:
            string = "Title: " + item['title'] + "\n" + "Summary: " + item['summary']
        else:
            string = "Summary: " + item['summary']
        summaries.append(string)    

docs = summaries
            
from sentence_transformers import SentenceTransformer   
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

topic_model = BERTopic(embedding_model=sentence_model)
topics, probs = topic_model.fit_transform(docs)

topic_labels = topic_model.generate_topic_labels(nr_words = 3, topic_prefix = False, word_length = 15, separator = " - ")
topic_model.set_topic_labels(topic_labels)

topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])

if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('summary_news_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('summary_news_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")

