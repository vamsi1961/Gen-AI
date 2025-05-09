import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError

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
index_name = "genai12345"
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SimpleField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Double)),  # Use SimpleField for vector data
    SearchableField(name="metadata", type=SearchFieldDataType.String)  # Add metadata field
]
index = SearchIndex(name=index_name, fields=fields)

# Create the index if it does not exist
index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key))
try:
    index_client.create_index(index)
except ResourceExistsError:
    print(f"Index '{index_name}' already exists.")

# Initialize the search client
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))

# Function to add documents with embeddings and metadata
def add_documents(documents):
    for doc in documents:
        doc['content_vector'] = generate_embeddings(doc['content'])
    search_client.upload_documents(documents)

# Function to retrieve a document by ID
def retrieve_document_by_id(doc_id):
    result = search_client.get_document(key=doc_id)
    return result

# Function to update a document
def update_document(doc_id, new_content, new_metadata):
    new_vector = generate_embeddings(new_content)
    updated_doc = {"id": doc_id, "content": new_content, "content_vector": new_vector, "metadata": new_metadata}
    search_client.merge_or_upload_documents([updated_doc])

# Function to delete a document
def delete_document(doc_id):
    search_client.delete_documents([{"id": doc_id}])

# Example usage
documents = [
    {"id": "1", "content": "This is a sample document.", "metadata": "Sample metadata"},
    {"id": "2", "content": "Another example document.", "metadata": "Example metadata"},
    {"id": "3", "content": "Yet another example document.", "metadata": "Additional metadata"}
]

# Add documents
add_documents(documents)

# Retrieve the document with ID "1"
document_id = "1"
document = retrieve_document_by_id(document_id)
print(f"Document with ID {document_id}:")
print(document)

# # Update a document
# update_document("1", "This is an updated sample document.", "Updated metadata")

# # Delete a document
# delete_document("2")
