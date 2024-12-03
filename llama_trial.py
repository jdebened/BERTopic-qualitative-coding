from torch import bfloat16
import transformers
from sentence_transformers import SentenceTransformer
from bertopic.representation import TextGeneration
from bertopic import BERTopic

# Define local model path
model_id = './llama2/Llama-2-7b-chat-hf'

# Quantization configuration
# bnb_config = transformers.BitsAndBytesConfig(
#    load_in_4bit=True,  # 4-bit quantization
#    bnb_4bit_quant_type='nf4',  # Normalized float 4
#    bnb_4bit_use_double_quant=True,  # Second quantization after the first
#    bnb_4bit_compute_dtype=bfloat16  # Computation type
#)

# Load tokenizer and model from local directory
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,  # Set this to False unless you have custom code locally
#    quantization_config=bnb_config,
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

docs = [
    "Artificial intelligence is transforming various industries.",
    "Machine learning focuses on pattern recognition.",
    "Deep learning utilizes neural networks.",
    "Natural language processing helps machines understand human language.",
    "Climate change affects the environment and causes extreme weather patterns.",
    "Renewable energy sources are essential in combating climate change.",
    "Environmental conservation efforts are critical for future generations.",
    "The economy is influenced by global events and market fluctuations.",
    "Stock markets are affected by interest rates and inflation rates.",
    "Investment in technology is essential for economic growth.",
    "AI technologies are reshaping healthcare, improving diagnosis, and treatment options.",
    "Machine learning algorithms are applied in fraud detection and risk assessment.",
    "Deep learning models achieve state-of-the-art results in image and speech recognition.",
    "NLP powers virtual assistants, chatbots, and translation tools for better communication.",
    "Global warming leads to rising sea levels and habitat destruction.",
    "Solar, wind, and hydropower are pivotal in reducing reliance on fossil fuels.",
    "Preserving biodiversity is vital for maintaining ecological balance.",
    "Economic policies play a significant role in shaping national and global markets.",
    "Cryptocurrencies and blockchain technology are disrupting traditional financial systems.",
    "Technological innovation fosters job creation and drives sustainable development.",
    "Artificial intelligence also impacts education by enabling personalized learning platforms.",
    "Machine learning is increasingly used in predictive analytics for business strategies.",
    "Deep learning applications extend to autonomous vehicles and smart home technologies.",
    "Natural language processing contributes to advancements in sentiment analysis and content moderation.",
    "Climate change mitigation requires global cooperation and stringent carbon regulations.",
    "Renewable energy advancements are crucial to achieving energy independence globally.",
    "Efforts to reduce deforestation directly support environmental conservation goals.",
    "Economic growth depends heavily on technological adaptation and innovation.",
    "Stock market volatility often reflects geopolitical tensions and policy changes.",
    "Investments in green technology are essential for achieving long-term sustainability.",
    "AI-driven automation is redefining workforce dynamics and operational efficiency.",
    "Machine learning enhances cybersecurity by detecting threats in real-time.",
    "Deep learning continues to revolutionize fields like healthcare and robotics.",
    "NLP applications are critical for multilingual support in global businesses.",
    "Climate action plans aim to transition industries toward sustainability and reduce emissions.",
    "Solar panels, wind turbines, and energy storage systems are transforming power grids.",
    "Biodiversity protection ensures resilience against ecosystem disruptions.",
    "Economic reforms influence trade agreements and international relations.",
    "Blockchain adoption is growing across sectors like finance, healthcare, and logistics.",
    "Innovation ecosystems thrive when governments and industries collaborate effectively.",
    "AI is revolutionizing content creation with tools for video editing and image generation.",
    "Machine learning aids in predictive maintenance for manufacturing industries.",
    "Deep learning powers facial recognition systems in modern security frameworks.",
    "Natural language processing improves customer service experiences through automation.",
    "Climate change impacts agriculture by altering growing seasons and crop yields.",
    "Wind energy farms are expanding as a key renewable energy source.",
    "Efficient water management systems are crucial in addressing climate challenges.",
    "Economic inequality is exacerbated by limited access to emerging technologies.",
    "Global trade policies affect the supply chain and international commerce.",
    "Cryptocurrency adoption is reshaping how transactions are conducted globally.",
    "AI enhances precision medicine by tailoring treatments to individual genetic profiles.",
    "Machine learning optimizes logistics and supply chain operations for companies.",
    "Deep learning is integral to advancements in prosthetics and assistive technologies.",
    "NLP supports sentiment analysis for tracking public opinion on social media.",
    "Rising temperatures contribute to increased frequency of wildfires worldwide.",
    "Hydrogen fuel cells are gaining traction as an alternative energy source.",
    "Carbon capture technologies aim to reduce greenhouse gas emissions effectively.",
    "Global financial crises often influence the stability of developing economies.",
    "E-commerce platforms rely on machine learning for personalized product recommendations.",
    "AI-powered analytics improve decision-making processes in corporate environments.",
    "Machine learning aids in medical imaging for early detection of diseases.",
    "Deep learning techniques are advancing natural disaster prediction models.",
    "NLP tools are critical for analyzing legal documents and simplifying workflows.",
    "Melting polar ice caps threaten marine ecosystems and global sea levels.",
    "Offshore wind farms are playing a growing role in renewable energy generation.",
    "Forest restoration initiatives help combat climate change and improve air quality.",
    "Economic globalization is transforming traditional job markets and employment patterns.",
    "Blockchain technology enhances transparency in supply chain management.",
    "AI-based fraud detection systems protect online banking and payment platforms.",
    "Machine learning aids in drug discovery, speeding up pharmaceutical research.",
    "Deep learning innovations drive advancements in personalized advertising strategies.",
    "NLP technologies enable accurate transcription and voice-to-text conversions.",
    "Urban heat islands worsen as cities expand without sustainable planning.",
    "Smart grids utilize renewable energy to optimize electricity distribution.",
    "Ecotourism promotes environmental awareness while supporting local economies.",
    "Economic diversification helps nations reduce reliance on single industries.",
    "Renewable energy investments are reducing dependency on fossil fuel imports.",
    "Autonomous drones powered by AI assist in disaster relief operations.",
    "Machine learning algorithms optimize energy consumption in smart buildings.",
    "Deep learning models enable real-time language translation in global communication."
]

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

# Define representation model
representation_model = {"Llama2": llama2}

# BERTopic setup
topic_model = BERTopic(
    representation_model=representation_model,
    top_n_words=10,
    verbose=True
)

# Train BERTopic model
topics, probs = topic_model.fit_transform(docs, embeddings)

# Adjust topic labels and export information
topic_info = topic_model.get_topic_info()
labels = [item[0] for item in topic_info["Llama2"]]
topic_model.set_topic_labels(labels)

topic_info = topic_model.get_topic_info()
document_info = topic_model.get_document_info(docs)
if 'Representative_Docs' in topic_info.columns:
    topic_info = topic_info.drop(columns=['Representative_Docs', 'Llama2'])

if 'Representative_Docs' in document_info.columns and 'Document' in document_info.columns:
    document_info = document_info.drop(columns=['Representative_Docs', 'Document'])
topic_info.to_csv('trial_topic_info.tsv', sep='\t', index=False)
document_info.to_csv('trial_document_info.tsv', sep='\t', index=False)

print("Topic information and document information have been saved to 'topic_info.tsv' and 'document_info.tsv'.")

