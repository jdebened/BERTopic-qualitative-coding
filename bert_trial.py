from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups

import os

def preprocess_single_file(file_path):
    data = []
    metadata_headers = ['Newsgroup:', 'Subject:', 'document_id:', 'Document_id:', 'From:']

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    document = []
    for line in lines:
        if any(header in line for header in metadata_headers):
            if document:  
                data.append(''.join(document))  
                document = []  
        else:
            document.append(line)

    if document:
        data.append(''.join(document))

    return data

def preprocess_directory(directory_path):
    data = []
    category_names = []
    categories = []

    files = sorted([f for f in os.listdir(directory_path) if f.endswith('.txt')])

    for group_index, file_name in enumerate(files):
        path = os.path.join(directory_path, file_name)

        file_data = preprocess_single_file(path)
        data.extend(file_data)  

        category_names.append(file_name.replace('.txt', ''))
        categories.extend([group_index + 1] * len(file_data))

    return data, categories, category_names

directory_path = './20_newsgroups/'  
docs, categories, category_names = preprocess_directory(directory_path)

print(len(docs), len(categories), len(category_names))


from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

topic_model = BERTopic(embedding_model=sentence_model)
topics, probs = topic_model.fit_transform(docs)

# print(topic_model.get_topic_info())

# print(topic_model.get_topic(0))

# print(topic_model.get_document_info(docs))

# topic_labels = topic_model.generate_topic_labels(nr_words = 3, topic_prefix = False, word_length = 15, separator = " - ")
# topic_model.set_topic_labels(topic_labels)

topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

# from transformers import pipeline
# from bertopic.representation import TextGeneration

# generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=50)

# Function to generate a label using the LLM
# def generate_label(keywords):
#    prompt = f"I have a topic described by the following keywords: {keywords}. Based on these keywords, what is a concise and descriptive label for this topic?"
#    response = generator(prompt)
#    return response[0]['generated_text']
# labels = []
# for topic_num in topic_info["Topic"]:
    # keywords = ', '.join([word for word, _ in topic_model.get_topic(topic_num)])
    # label = generate_label(keywords)
    # labels.append(label)
#    if topic_num == -1:  # Skip outliers
#        labels.append("Outliers")
#    else:
#        keywords = ', '.join([word for word, _ in topic_model.get_topic(topic_num)[:5]])
#        label = generate_label(keywords)
#        labels.append(label)

# topic_model.set_topic_labels(labels)
# topic_info = topic_model.get_topic_info()
if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])


if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('llm_bert_trial_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('llm_bert_trial_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")
