# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_vector_search.py
DESCRIPTION:
    This sample demonstrates how to get search results from a basic search text
    from an Azure Search index.
USAGE:
    python sample_vector_search.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_SEARCH_SERVICE_ENDPOINT - the endpoint of your Azure Cognitive Search service
    2) AZURE_SEARCH_INDEX_NAME - the name of your search index (e.g. "hotels-sample-index")
    3) AZURE_SEARCH_API_KEY - your search API key
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery



index_name = "hotels-sample-index"


def get_embeddings(text: str):
    # There are a few ways to get embeddings. This is just one example.
    import openai

        

    response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")

    return response.data[0].embedding

def get_hotel_index(name: str):
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )

    fields = [
        SimpleField(name="hotelId", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="hotelName",
            type=SearchFieldDataType.String,
            sortable=True,
            filterable=True,
        ),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SearchField(
            name="descriptionVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config",
        ),
        SearchableField(
            name="category",
            type=SearchFieldDataType.String,
            sortable=True,
            filterable=True,
            facetable=True,
        ),
    ]
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)

def get_hotel_documents():
    docs = [
        {
            "hotelId": "1",
            "hotelName": "Fancy Stay",
            "description": "Best hotel in town if you like luxury hotels.",
            "descriptionVector": get_embeddings("Best hotel in town if you like luxury hotels."),
            "category": "Luxury",
        },
        {
            "hotelId": "2",
            "hotelName": "Roach Motel",
            "description": "Cheapest hotel in town. Infact, a motel.",
            "descriptionVector": get_embeddings("Cheapest hotel in town. Infact, a motel."),
            "category": "Budget",
        },
        {
            "hotelId": "3",
            "hotelName": "EconoStay",
            "description": "Very popular hotel in town.",
            "descriptionVector": get_embeddings("Very popular hotel in town."),
            "category": "Budget",
        },
        {
            "hotelId": "4",
            "hotelName": "Modern Stay",
            "description": "Modern architecture, very polite staff and very clean. Also very affordable.",
            "descriptionVector": get_embeddings(
                "Modern architecture, very polite staff and very clean. Also very affordable."
            ),
            "category": "Luxury",
        },
        {
            "hotelId": "5",
            "hotelName": "Secret Point",
            "description": "One of the best hotel in town. The hotel is ideally located on the main commercial artery of the city in the heart of New York.",
            "descriptionVector": get_embeddings(
                "One of the best hotel in town. The hotel is ideally located on the main commercial artery of the city in the heart of New York."
            ),
            "category": "Boutique",
        },
    ]
    return docs

def single_vector_search():
    # [START single_vector_search]
    query = "Top hotels in town"

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=2, fields="descriptionVector")

    results = search_client.search(
        vector_queries=[vector_query],
        select=["hotelId", "hotelName"],
    )

    for result in results:
        print(result)
    # [END single_vector_search]

def single_vector_search_with_filter():
    # [START single_vector_search_with_filter]
    query = "Top hotels in town"

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=2, fields="descriptionVector")

    results = search_client.search(
        search_text=query,
        filter="category eq 'Luxury'",
        select=["hotelId", "hotelName"],
        top = 2
    )

    for result in results:
        print(result)
    # [END single_vector_search_with_filter]

def simple_hybrid_search():
    # [START simple_hybrid_search]
    query = "Top hotels in town"

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=2, fields="descriptionVector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["hotelId", "hotelName"],
        top=2
    )

    for result in results:
        print(result)
    # [END simple_hybrid_search]

def multi_query_hybrid_search(queries):
    # [START multi_query_hybrid_search]
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    
    all_results = []
    
    for query in queries:
        # Create a vectorized query for the current query
        vector_query = VectorizedQuery(
            vector=get_embeddings(query), 
            k_nearest_neighbors=2, 
            fields="descriptionVector"
        )
        
        # Execute the search for this query
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["hotelId", "hotelName"],
            top=2
        )
        
        # Store results for this query
        query_results = list(results)
        all_results.append({
            "query": query,
            "results": query_results
        })
        
        # Print results for this query
        print(f"Results for query: '{query}'")
        for result in query_results:
            print(result)
        print("---")
    
    return all_results
    # [END multi_query_hybrid_search]

def get_all_search_indexes():
    # Create a SearchIndexClient
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(service_endpoint, credential)
    
    # Get all indexes
    indexes = list(index_client.list_indexes())

    final_indexes = [index.name for index in indexes]
        
    return final_indexes


def update_hotel_documents():
    # Get the existing hotel documents that need to be updated
    updated_docs = [
        {
            "hotelId": "1",
            "hotelName": "Fancy Stay Deluxe",  # Updated name
            "description": "Best hotel in town if you like luxury hotels. Newly renovated with spa services.",  # Updated description
            "descriptionVector": get_embeddings("Best hotel in town if you like luxury hotels. Newly renovated with spa services."),
            "category": "Luxury",
        },
        {
            "hotelId": "2",
            "hotelName": "Roach Motel",
            "description": "Cheapest hotel in town. In fact, a motel with new management.",  # Corrected spelling and updated
            "descriptionVector": get_embeddings("Cheapest hotel in town. In fact, a motel with new management."),
            "category": "Budget",
        },
        {
            "hotelId": "3",
            "hotelName": "EconoStay Plus",  # Updated name
            "description": "Very popular hotel in town with free breakfast.",  # Updated description
            "descriptionVector": get_embeddings("Very popular hotel in town with free breakfast."),
            "category": "Budget",
        },
        {
            "hotelId": "4",
            "hotelName": "Modern Stay",
            "description": "Modern architecture, very polite staff and very clean. Also very affordable with gym access.",  # Updated description
            "descriptionVector": get_embeddings("Modern architecture, very polite staff and very clean. Also very affordable with gym access."),
            "category": "Luxury",
        },
        {
            "hotelId": "5",
            "hotelName": "Secret Point Boutique",  # Updated name
            "description": "One of the best hotels in town. The hotel is ideally located on the main commercial artery of the city in the heart of New York. Walking distance to attractions.",  # Updated description
            "descriptionVector": get_embeddings("One of the best hotels in town. The hotel is ideally located on the main commercial artery of the city in the heart of New York. Walking distance to attractions."),
            "category": "Boutique",
        },
    ]

def get_documents_by_id(document_ids):
    """
    Retrieves documents by hotelId assuming hotelId is the key field.
    """
    try:
        credential = AzureKeyCredential(key)
        search_client = SearchClient(service_endpoint, index_name, credential)

        documents = []
        for doc_id in document_ids:
            try:
                doc = search_client.get_document(key=doc_id)
                documents.append(doc)
                print(f"  - Hotel ID: {doc['hotelId']}, Name: {doc['hotelName']}")
            except Exception as e:
                print(f"  - Could not find document with ID {doc_id}: {e}")

        print(f"Retrieved {len(documents)} document(s) out of {len(document_ids)} requested.")
        return documents

    except Exception as ex:
        print(f"Error: {ex}")
        raise

def get_all_documents():
    """
    Retrieves all documents from the Azure Cognitive Search index.
    """
    try:
        credential = AzureKeyCredential(key)
        search_client = SearchClient(service_endpoint, index_name, credential)

        # search_text="*" matches all documents
        results = search_client.search(
            search_text="*",
            select=["hotelId", "hotelName", "description", "category"],  # Adjust fields as needed
            top=1000  # Optional: page size per request (default is 50)
        )

        documents = list(results)

        print(f"Retrieved {len(documents)} document(s).")
        for doc in documents:
            print(f"  - Hotel ID: {doc['hotelId']}, Name: {doc['hotelName']}")

        return documents

    except Exception as ex:
        print(f"Error retrieving documents: {ex}")
        raise

if __name__ == "__main__":
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(service_endpoint, credential)
    index = get_hotel_index(index_name)
    
    if not index_client.get_index(index.name):
        index_client.create_index(index)

    # index_client.create_index(index)
    client = SearchClient(service_endpoint, index_name, credential)
    hotel_docs = get_hotel_documents()

    # create embeddings
    client.upload_documents(documents=hotel_docs)


    # get al indexes
    all_indexes = get_all_search_indexes()

    # upsert it is same as create
    result = client.upload_documents(documents=hotel_docs)

    # get the embeddings given id
    document_ids = ["1", "3", "5"]
    retrieved_docs = get_documents_by_id(document_ids)


    # get all documents
    print(get_all_documents())


    def delete_document_by_id(doc_id):
        credential = AzureKeyCredential(key)
        search_client = SearchClient(service_endpoint, index_name, credential)

        result = search_client.delete_documents(documents=[{"@search.action": "delete", "hotelId": doc_id}])
        print(f"Deleted document with ID: {doc_id}")

    def delete_documents_by_ids(doc_ids):
        credential = AzureKeyCredential(key)
        search_client = SearchClient(service_endpoint, index_name, credential)

        docs_to_delete = [{"@search.action": "delete", "hotelId": doc_id} for doc_id in doc_ids]
        result = search_client.upload_documents(documents=docs_to_delete)
        
        print(f"Requested deletion of {len(doc_ids)} document(s).")
        for r in result:
            print(f"  - ID: {r['key']}, Status: {r['statusCode']}")

    def delete_index(index_name_to_delete):
        credential = AzureKeyCredential(key)
        index_client = SearchIndexClient(service_endpoint, credential)

        index_client.delete_index(index_name_to_delete)
        print(f"Deleted index: {index_name_to_delete}")


    # update the documents


    print("single vector search")

    # single_vector_search()
    print("single vector search with filter")
    single_vector_search_with_filter()
    print("simple hybrid search")
    # simple_hybrid_search()

    queries = [
    "Top hotels in town",
    "Hotels with swimming pools",
    "Pet-friendly accommodations",
    "Luxury suites downtown",
    "Budget hotels near airport"
]

    results = multi_query_hybrid_search(queries)
