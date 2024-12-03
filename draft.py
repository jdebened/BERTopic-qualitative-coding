from sentence_transformers import SentenceTransformer

# Load your local model
model_path = './all-MiniLM-L6-v2'
model = SentenceTransformer(model_path, device='cpu')

# Test encoding on a sample document
sample_docs = ["This is a test document."]
embeddings = model.encode(sample_docs)

print(embeddings)

