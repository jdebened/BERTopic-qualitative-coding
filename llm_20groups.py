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

model_path = './all-MiniLM-L6-v2'  
sentence_model = SentenceTransformer(model_path)
embeddings = sentence_model.encode(docs, show_progress_bar=True)

from transformers import pipeline
from bertopic.representation import TextGeneration

# prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"
prompt = """
I have a topic described by the following keywords: '[KEYWORDS]'. 
Based on the information about the topic above, please create a short label of this topic. Make sure you return only the label. Use consistent structure and noun-based phrasing.
"""
# prompt = (
#    "Given these keywords: [KEYWORDS], summarize the topic in a short, descriptive phrase. "
#    "The phrase should be meaningful and capture the essence of the topic."
# )

# Create your representation model
# generator = pipeline('text2text-generation', model='google/flan-t5-base', device='cuda', max_length=20, min_length=5, num_return_sequences=1, do_sample=True, top_k = 50, top_p = 0.9)

generator = pipeline('text2text-generation', model='google/flan-t5-base', device='cuda')
# generator = pipeline('text2text-generation', model='google/flan-t5-base')
# representation_model = TextGeneration(generator, prompt=prompt)

system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

example_prompt = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""

main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""
# prompt = system_prompt + example_prompt + main_prompt
system_prompt_shorten = """
<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant for labeling topics. 
<</SYS>>
"""
example_prompt_shorten = """
I have a topic that contains some documents.  
The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. 
Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""
main_prompt_shorten = """
[INST]
I have a topic that contains some documents.  
The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. 
Make sure you to only return the label and nothing more.

[/INST]
"""
# prompt = system_prompt_shorten + example_prompt_shorten + main_prompt_shorten



representation_model = {
    "FlanT5 - Base": TextGeneration(generator, prompt=prompt)
}

# topic_model = BERTopic(representation_model=representation_model, embedding_model=sentence_model)
topic_model = BERTopic(embedding_model=sentence_model,
                       representation_model=representation_model,
                       # Hyperparameters
                       top_n_words=10,
                       verbose=True)
topics, probs = topic_model.fit_transform(docs, embeddings)

# print(topic_model.get_topic_info())

# print(topic_model.get_topic(0))

# print(topic_model.get_document_info(docs))

topic_info = topic_model.get_topic_info()

labels = [item[0] for item in topic_info["FlanT5 - Base"]]
topic_model.set_topic_labels(labels)
topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)
if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])


if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('./llm_label/llm_20groups_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('./llm_label/llm_20groups_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")
