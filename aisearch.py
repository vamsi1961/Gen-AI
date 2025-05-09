import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)
from azure.core.credentials import AzureKeyCredential

# Set up Azure OpenAI API key and endpoint
openai.api_type = "azure"
openai.api_base = "https://genaitcgazuregpt.openai.azure.com"
openai.api_version = "2024-02-15-preview"
openai.api_key = ""

# Function to generate embeddings using Azure OpenAI
def generate_embeddings(text):
    response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Set up Azure Search credentials and endpoint
search_endpoint = "https://genai-azureaisearch.search.windows.net"
search_api_key = ""

# Define the index schema
index_name = "genai1234"
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SimpleField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Double))  # Use SimpleField for vector data
]
index = SearchIndex(name=index_name, fields=fields)

# Prepare and upload documents with embeddings
documents = [
    {"id": "1", "content": "This is a sample document.", "content_vector": generate_embeddings("This is a sample document.")},
    {"id": "2", "content": "Another example document.", "content_vector": generate_embeddings("Another example document.")},
    {"id": "3", "content": "Another example document.", "content_vector": generate_embeddings("Another example document.")}
]
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))
search_client.upload_documents(documents)

# Perform vector search
query_vector = generate_embeddings("sample search query")
search_query = {
    "search": "*",
    "vector": {
        "value": query_vector,
        "fields": ["content_vector"]
    }
}
results = search_client.search(search_query)
for result in results:
    print(result)
