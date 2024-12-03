from bertopic import BERTopic
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

import transformers
from bertopic.representation import TextGeneration

model_id = './llama2/Llama-2-7b-chat-hf'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,  # Set this to False unless you have custom code locally
    device_map='auto',
    local_files_only=True
)

model.eval()

# Define the generator pipeline
generator = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)
from sentence_transformers import SentenceTransformer
# Pre-calculate embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# System and prompt templates
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""
example_prompt = """
[INST]
I have a topic that contains the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Environmental impacts of eating meat
"""
main_prompt = """
[INST]
I have a topic that contains the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

# Text generation configuration
prompt = system_prompt + example_prompt + main_prompt
llama2 = TextGeneration(generator, prompt=prompt)
representation_model = {"Llama2": llama2}

# topic_model = BERTopic(representation_model=representation_model, embedding_model=sentence_model)
topic_model = BERTopic(embedding_model=sentence_model,
                       representation_model=representation_model,
                       # Hyperparameters
                       top_n_words=10,
                       verbose=True)
topics, probs = topic_model.fit_transform(docs, embeddings)

topic_info = topic_model.get_topic_info()
labels = [item[0] for item in topic_info["Llama2"]]
topic_model.set_topic_labels(labels)

topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)

if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs'])

if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('./llama_label/llm_20groups_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('./llama_label/llm_20groups_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")
