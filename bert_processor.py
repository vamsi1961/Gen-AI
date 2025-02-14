"""
========================================================================================
BERT Processor: Custom Natural Language Inference (NLI) and Response Evaluation Module
========================================================================================

Description:
------------
This module implements a comprehensive framework for Natural Language Inference (NLI) and
response evaluation using a custom model based on `roberta-large-mnli`. It supports tasks
such as entailment classification, similarity analysis, response validation, embedding
generation, and integration with external systems like Excel for persistence and reporting.

Main Features:
--------------
1. **Custom NLI Model**:
   - Extends `roberta-large-mnli` with additional layers for multi-class entailment classification.
   - Outputs entailment relationships: "match," "partial match," or "no match."

2. **Entailment Analysis**:
   - Evaluates semantic relationships between question-answer pairs and responses.
   - Provides entailment scores, similarity metrics, and contextual insights.

3. **Response Evaluation**:
   - Validates responses against ground truth or reference answers.
   - Analyzes contextual and semantic accuracy.

4. **Embedding Generation**:
   - Generates embeddings for text data to enable similarity analysis and retrieval.

5. **Similarity Retrieval**:
   - Uses cosine similarity to find semantically related entries in the dataset.

6. **Persistence and Reusability**:
   - Saves model weights and entailment records for reuse and incremental updates.

Usage Examples:
---------------
1. **Analyze Entailment**:
    ```python
    result = analyze_entailment_with_bert("What is AI?", "AI is artificial intelligence.")
    print(result["entailment_label"])  # Output: "match"
    ```

2. **Evaluate Responses**:
    ```python
    result = evaluate_llm_response(
        "How do I reset my router?",
        "Technical support inquiry",
        "Turn it off for 30 seconds and restart."
    )
    print(result["entailment_label"])  # Output: "match"
    ```

3. **Generate Contextual Response**:
    ```python
    context = generateContextResponseFromBert("Why is my internet slow?", "Check your router.", "Verify connection.")
    print(context)
    ```

4. **Retrieve Similar Entries**:
    ```python
    similar_entries = retrieve_similar_nli("What causes rainfall?", bert_entailments, threshold=0.8)
    print(similar_entries)
    ```

5. **Save Model and Data**:
    ```python
    save_model_and_data(new_entailment_data)
    ```

Dependencies:
-------------
- `transformers`: For tokenizer and model components.
- `torch`: For model definition and inference.
- `numpy`, `sklearn`: For similarity metrics.
- `save_to_excel`: Custom module for saving outputs.

Author:
-------
[Your Name]

Date:
-----
[Insert Date]
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import os
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import unicodedata

from save_to_excel import storeQuestionAndAnswer

# Path to save the custom model's state
model_path = "custom_nli_model.pth"
entailment_path = "saved_entailments.json"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# Define Custom Model Class
class CustomNLIModel(nn.Module):
    def __init__(self):
        super(CustomNLIModel, self).__init__()
        self.bert = AutoModel.from_pretrained("roberta-large-mnli")
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),  # Custom linear layer
            nn.ReLU(),
            nn.Linear(128, 3)  # Output layer for entailment, neutral, contradiction
        )
        self.stored_data = []  # Initialize stored_data as an empty list
    def forward(self, input_ids, attention_mask):
        # Get the output from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the output to feed into our custom classifier
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # [CLS] token representation
        return logits

# Function to create default model weights and initialize stored data
def create_default_model_weights(model_path):
    print(f"Bert_processing.py:Startup:create_default_model_weights: Creating a default model state file...")
    # Initialize a default model
    custom_model = CustomNLIModel()  # Ensure CustomNLIModel is defined elsewhere in your script
    custom_model.stored_data = []  # Initialize empty stored data

    # Save the model weights and stored data to a .pth file
    torch.save({
        "model_state_dict": custom_model.state_dict(),
        "stored_data": custom_model.stored_data  # Save empty stored data
    }, model_path)

    print(f"Bert_processing.py:Startup:Default model weights and stored data saved to {model_path}")
    
# Instantiate Custom Model
custom_model = CustomNLIModel()

# Ensure the custom model's state file exists
if not os.path.exists(model_path):
    print(f"Bert_processing.py:Startup:{model_path} does not exist. Creating a default model state file...")
    create_default_model_weights(model_path)
else:
    print(f"Bert_processing.py:Startup:{model_path} already exists.")

# Ensure the entailments file exists
if not os.path.exists(entailment_path):
    print(f"Bert_processing.py:Startup:{entailment_path} does not exist. Creating a default entailments file...")
    with open(entailment_path, 'w') as f:
        json.dump([], f)  # Save an empty dictionary as the default content
        print(f"Bert_processing.py:Startup:Default entailments file created at {entailment_path}")
else:
    print(f"Bert_processing.py:Startup:{entailment_path} already exists.")

# Load the trained custom model's state and stored data
if os.path.exists(model_path):
    print("Bert_processing.py:Startup:Load the trained custom model's state and stored data")
    checkpoint = torch.load(model_path, weights_only=True)  # Load the checkpoint

    # Load the model's weights
    custom_model.load_state_dict(checkpoint["model_state_dict"])

    # Check if the entailment_path file exists
    if os.path.exists(entailment_path):
        try:
            with open(entailment_path, "r") as f:
                # Attempt to load the JSON content
                content = json.load(f)
                
                # Check if the file content is an empty dictionary
                if isinstance(content, dict) and not content:
                    print("Bert_processing.py:Startup:Stored data file contains an empty dictionary. Initializing empty stored_data.")
                    custom_model.stored_data = []  # Initialize as an empty list
                else:
                    # Assume valid stored data and load it
                    custom_model.stored_data = content
                    print("Bert_processing.py:Startup:Stored data loaded successfully.")
        except json.JSONDecodeError:
            # Handle invalid JSON content
            print("Bert_processing.py:Startup:Stored data file contains invalid JSON. Initializing empty stored_data.")
            custom_model.stored_data = []  # Initialize as an empty list
    else:
        # If the file does not exist, initialize empty stored data
        print("Bert_processing.py:Startup:Stored data file not found. Initializing empty stored_data.")
        custom_model.stored_data = []  # Initialize as an empty list

        custom_model.eval()
        
        print("Bert_processing.py:Startup:Custom model loaded successfully.")
else:
    # If the model file does not exist, notify the user
    custom_model.stored_data = []
    print("Bert_processing.py:Startup:Custom model file not found. Please train and save the model.")
#
#
#
def load_saved_entailments(file_path="saved_entailments.json"):
    """
    Load saved entailments from a JSON file. If the file is empty, malformed, or does not exist,
    initialize it as an empty list.
    """
    print(f"Bert_processing.py:Startup:load_saved_entailments:Function called!")
    entailments = []

    if os.path.exists(file_path):
        try:
            # Check if the file is empty
            if os.stat(file_path).st_size == 0:
                print(f"Bert_processing.py:Startup:load_saved_entailments: {file_path} is empty. Initializing with an empty list.")
                return entailments  # Return an empty list for an empty file

            # Attempt to load JSON data from the file
            with open(file_path, "r", encoding="utf-8") as f:
                # Use `json.load` instead of line-by-line to handle arrays properly
                entailments = json.load(f)

            # Verify the loaded data is a list
            if not isinstance(entailments, list):
                print(f"Bert_processing.py:Startup:load_saved_entailments:Invalid data format in {file_path}. Expected a list. Reinitializing.")
                return []

            print(f"Bert_processing.py:Startup:load_saved_entailments:Loaded {len(entailments)} saved entailments.")
        except json.JSONDecodeError as e:
            print(f"Bert_processing.py:Startup:load_saved_entailments:Error decoding JSON in {file_path}: {e}. Initializing with an empty list.")
            return []
        except Exception as e:
            print(f"Bert_processing.py:Startup:load_saved_entailments:Unexpected error reading {file_path}: {e}. Initializing with an empty list.")
            return []
    else:
        print(f"Bert_processing.py:Startup:load_saved_entailments: {file_path} not found. Initializing a new file.")
        # Create the file with an empty JSON list
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)  # Start with an empty JSON array
        except Exception as e:
            print(f"Bert_processing.py:Startup:load_saved_entailments:Error creating {file_path}: {e}")

    return entailments


# Load entailments on startup
bert_entailments = load_saved_entailments()
#
#
#
print(f"Bert_processing.py:Startup: Number of records in stored_data: {len(custom_model.stored_data)}")
#
# Internal call to evaluate semantic relationships
#
def generate_entailment_from_bertOLD(question, answer, golden_response):
    """
    Performs Natural Language Inference (NLI) to evaluate semantic relationships:
    1. Between a question and an answer (LLM-generated response).
    2. Optionally, between the answer (LLM-generated response) and a provided reference response.

    Parameters:
        question (str): The input question or premise text.
        answer (str): The LLM-generated response or hypothesis text.
        golden_response (str, optional): A reference or gold-standard response for comparison.
        entailment_engine (str): The engine used for entailment analysis. Default is "bert".

    Returns:
        dict: Contains the following:
            - "question": The input question.
            - "answer": The generated response.
            - "golden_response": The optional reference response.
            - "entailment_label": The entailment result for the question and answer.
            - "contradiction_prob": The probability of contradiction for the question and answer.
            - "neutral_prob": The probability of neutrality for the question and answer.
            - "entailment_prob": The probability of entailment for the question and answer.
            - "provided_response_score": The entailment probability for the answer and provided response (if provided).

    Raises:
        ValueError: If the specified entailment engine is not supported.
    """
   
    print(f"bert_processor.py:generate_entailment_from_bert:Entailment Engine is BERT")
    # Step 1: Tokenize and encode the question and answer
    print(f"bert_processor.py:generate_entailment_from_bert:Step 1 tokenize question and answer: {question}: {answer}")
    inputs = tokenizer(question, answer, return_tensors="pt", truncation=True, padding=True)

    # Step 2: Perform inference using the BERT model
    outputs = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs
    print(f"bert_processor.py:generate_entailment_from_bert:Step 2 perform inference using BERT Model: {logits}")

    # Step 3: Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=1).detach().numpy()
    print(f"bert_processor.py:generate_entailment_from_bert:Step 3: Convert logits to probabilities using softmax: {probabilities}")
    
    # Step 4: Define entailment labels
    labels = ["match", "partial_match", "no_match"]

    # Step 5: Extract probabilities for entailment, neutrality, and contradiction
    entailment_prob = probabilities[0][2]  # Probability of entailment
    neutral_prob = probabilities[0][1]     # Probability of neutrality
    contradiction_prob = probabilities[0][0]  # Probability of contradiction
    print(f"bert_processor.py:generate_entailment_from_bert:Step 5: Extract probabilities for entailment, neutrality, and contradiction:entailment_prob {entailment_prob}")
    print(f"bert_processor.py:generate_entailment_from_bert:Step 5: Extract probabilities for entailment, neutrality, and contradiction:neutral_prob {neutral_prob}")
    print(f"bert_processor.py:generate_entailment_from_bert:Step 5: Extract probabilities for entailment, neutrality, and contradiction:contradiction_prob {contradiction_prob}")
    # Step 6: Determine the entailment label based on probabilities
    if entailment_prob > 0.8:
        entailment_label = "match"
    elif neutral_prob > 0.5:
        entailment_label = "partial_match"
    else:
        entailment_label = "no_match"
        
    print(f"bert_processor.py:generate_entailment_from_bert:Step 6: Determine the entailment label based on probabilities:contradiction_prob:entailment_label: {entailment_label}")
    # Step 7: Process the provided response with the answer if the provided response is given
    provided_response_score = None
    provided_entailment_label = None
    print(f"bert_processor.py:generate_entailment_from_bert:Step 7: Process the provided response: {golden_response}")
    if golden_response:
        print(f"bert_processor.py:generate_entailment_from_bert:Step 7: Process the provided response with the answer if the provided response is given:golden_response: {golden_response}")
        # Compare the generated answer with the provided (gold standard) response
        inputs_provided = tokenizer(answer, golden_response, return_tensors="pt", truncation=True, padding=True)
        outputs_provided = custom_model(input_ids=inputs_provided['input_ids'], attention_mask=inputs_provided['attention_mask'])
        probabilities_provided = torch.softmax(outputs_provided, dim=1).detach().numpy()

        # Calculate the entailment probability
        provided_response_score = probabilities_provided[0][2]  # Probability of entailment for answer vs. gold standard

        # Determine the entailment label
        if provided_response_score > 0.8:
            provided_entailment_label = "match"
        elif probabilities_provided[0][1] > 0.5:
            provided_entailment_label = "partial_match"
        else:
            provided_entailment_label = "no_match"
        print(f"bert_processor.py:generate_entailment_from_bert:Step 7: Process the provided response with the answer if the provided response is given:provided_entailment_label: {provided_entailment_label}")
        print(f"bert_processor.py:generate_entailment_from_bert:Step 7: Process the provided response with the answer if the provided response is given:provided_response_score: {provided_response_score}")
        print(f"bert_processor.py:generate_entailment_from_bert:Step 7: Process the provided response with the answer if the provided response is given:provided_response: {golden_response}")
        print(f"bert_processor.py:generate_entailment_from_bert:Step 7: Process the provided response with the answer if the provided response is given:probabilities_provided: {probabilities_provided}")
    # Step 8: Construct the result dictionary
    print(f"bert_processor.py:generate_entailment_from_bert:Step 8: Building the result list")
    result = {
        "question": question,  # Input question
        "answer": answer,  # Generated answer
        "provided_response": golden_response,  # Provided (gold standard) response
        "entailment_label": entailment_label,  # Question vs. Answer comparison
        "contradiction_prob": float(contradiction_prob),  # Question vs. Answer
        "neutral_prob": float(neutral_prob),  # Question vs. Answer
        "entailment_prob": float(entailment_prob),  # Question vs. Answer
        "provided_response_score": float(provided_response_score) if golden_response else None,  # Answer vs. Gold Standard
        "provided_entailment_label": provided_entailment_label  # Answer vs. Gold Standard
    }

    # Step 9: Save entailment data into a persistent structure
    print(f"bert_processor.py:generate_entailment_from_bert:Step 9: Building the entailment_data")
    entailment_data = {
        "inputs": {
            "question": question,  # Input question
            "answer": answer,  # Generated answer
            "provided_response": golden_response  # Provided (gold standard) response
        },
        "label": entailment_label,  # Question vs. Answer
        "scores": {
            "contradiction": float(contradiction_prob),  # Question vs. Answer
            "neutral": float(neutral_prob),  # Question vs. Answer
            "entailment": float(entailment_prob),  # Question vs. Answer
            "provided_response_score": float(provided_response_score) if provided_response_score else None  # Answer vs. Gold Standard
        }
    }
    print("bert_processor.py:generate_entailment_from_bert:Saving entailment data:", entailment_data)
    save_model_and_data(entailment_data)

    # Step 10: Return the result dictionary
    return result
#
#
#
def retrieve_similar_nli(query, db_embeddings, threshold=0.9, top_k=5):
    """
    Retrieve similar entries from the BERT database based on semantic similarity.
    
    Args:
        query (str): The query text (e.g., a question) to search for similar entries.
        db_embeddings (list): A list of stored embeddings in the BERT database.
        threshold (float): Minimum similarity score to consider a result as similar.
        top_k (int): Maximum number of similar results to return.

    Returns:
        list: A list of tuples containing indices and similarity scores of similar entries.
    """
    # Generate embedding for the query
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = custom_model.bert(**inputs).last_hidden_state[:, 0, :].numpy()
    
    # Compute cosine similarity between query embedding and database embeddings
    similarities = cosine_similarity(query_embedding, np.array(db_embeddings))[0]
    
    # Find indices and scores of top_k similar results above the threshold
    similar_indices = [(idx, score) for idx, score in enumerate(similarities) if score >= threshold]
    similar_indices = sorted(similar_indices, key=lambda x: x[1], reverse=True)[:top_k]
    
    return similar_indices
#
#
#
def evaluate_llm_response(question, context, llm_response, provided_response=None, entailment_engine="bert"):
    """
    Evaluate an LLM response against a question (with context) and optionally a provided reference response.

    Parameters:
        question (str): The question from the user.
        context (str): Additional context provided with the question.
        llm_response (str): The response generated by the LLM.
        provided_response (str, optional): A gold-standard or retrieved answer for comparison.
        entailment_engine (str): The entailment engine to use (default is "bert").

    Returns:
        dict: Evaluation results including entailment labels, probabilities, and similarities.
    """
    if entailment_engine == 'bert':
        # Combine question and context for evaluation
        question_with_context = f"{context} {question}" if context else question

        # Step 1: Compare Question + Context with LLM Response
        inputs = tokenizer(question_with_context, llm_response, return_tensors="pt", truncation=True, padding=True)
        outputs = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probabilities = torch.softmax(outputs, dim=1).detach().numpy()

        # Define entailment labels and thresholds
        labels = ["no_match", "partial_match", "match"]
        entailment_prob = probabilities[0][2]  # Entailment (match) probability
        neutral_prob = probabilities[0][1]  # Neutral (partial match) probability

        if entailment_prob > 0.8:
            entailment_label = "match"
        elif neutral_prob > 0.5:
            entailment_label = "partial_match"
        else:
            entailment_label = "no_match"

        # Step 2: Compare LLM Response with Provided Response (if available)
        provided_response_score = None
        provided_entailment_label = None
        if provided_response:
            inputs_provided = tokenizer(llm_response, provided_response, return_tensors="pt", truncation=True, padding=True)
            outputs_provided = custom_model(input_ids=inputs_provided['input_ids'], attention_mask=inputs_provided['attention_mask'])
            probabilities_provided = torch.softmax(outputs_provided, dim=1).detach().numpy()

            # Calculate entailment probability and label for provided response
            provided_response_score = probabilities_provided[0][2]  # Entailment (match) probability for provided response
            if provided_response_score > 0.8:
                provided_entailment_label = "match"
            elif probabilities_provided[0][1] > 0.5:
                provided_entailment_label = "partial_match"
            else:
                provided_entailment_label = "no_match"

        # Return evaluation results
        result = {
            "question_with_context": question_with_context,
            "llm_response": llm_response,
            "provided_response": provided_response,
            "entailment_label": entailment_label,  # Question + Context vs. LLM Response
            "entailment_prob": float(entailment_prob),
            "neutral_prob": float(neutral_prob),
            "provided_response_score": float(provided_response_score) if provided_response else None,
            "provided_entailment_label": provided_entailment_label  # LLM Response vs. Provided Response
        }

        return result

    else:
        raise ValueError(f"Unsupported entailment engine: {entailment_engine}")
#
#
#
def save_model_and_dataOLD(entailment_data):
    """
    Save the updated model weights and entailment data to disk.
    """
    print("bert_processor.py:save_model_and_data: Function called!")
    
    global custom_model
    global model_path
    global entailment_path
    
    if not entailment_data:
        print("bert_processor.py:save_model_and_data: No entailment data to save!")
        return

    print("bert_processor.py:save_model_and_data: Entailment data:", entailment_data)
    
    # Ensure the custom model is available
    if custom_model is None:
        print("Error: custom_model is None. Ensure the model is initialized and trained.")
        return

    # Check if the entailment file exists, and initialize it if necessary
    if not os.path.exists(entailment_path):
        print(f"bert_processor.py:save_model_and_data: {entailment_path} not found. Creating a new file.")
        with open(entailment_path, "w") as f:
            json.dump([], f)  # Initialize the file with an empty list

    # Load existing entailments from the file
    try:
        with open(entailment_path, "r") as f:
            try:
                entailments = json.load(f)
                if not isinstance(entailments, list):
                    raise ValueError("Entailment data should be a list.")
            except json.JSONDecodeError:
                print("bert_processor.py:save_model_and_data: Error decoding JSON. Reinitializing as an empty list.")
                entailments = []
    except Exception as e:
        print(f"bert_processor.py:save_model_and_data: Error loading entailments: {e}. Reinitializing as an empty list.")
        entailments = []

    # Append the new entailment data
    entailments.append(entailment_data)

    # Save the updated entailments back to the file
    try:
        with open(entailment_path, "w") as f:
            json.dump(entailments, f, indent=4)
        print(f"bert_processor.py:save_model_and_data: Saved {len(entailments)} entailments to {entailment_path}.")
    except Exception as e:
        print(f"bert_processor.py:save_model_and_data: Error saving entailment data: {e}")
        return

    # Save the model's updated weights
    try:
        torch.save(custom_model.state_dict(), model_path)
        print(f"bert_processor.py:save_model_and_data: Model and entailments saved: {model_path}, {entailment_path}")
    except Exception as e:
        print(f"bert_processor.py:save_model_and_data: Error saving model weights: {e}")
        
#
#
#
def findAnswerFromBertModel(question):
    """
    Uses the loaded BERT model and stored data to find the best matching question
    and returns the associated answer.

    Parameters:
        question (str): The input question provided by the user.

    Returns:
        str: The answer associated with the matched question, or an empty string if no match is found.
    """
    print("bert_processor.py:findAnswerFromBertModel:Function called!")

    # Ensure the custom_model has stored_data initialized
    if not hasattr(custom_model, "stored_data") or not isinstance(custom_model.stored_data, list):
        print("bert_processor.py:findAnswerFromBertModel:No stored data available in custom_model.")
        return ""

    # Generate the embedding for the new question
    try:
        question_embedding = bert_generate_embedding(question)
        question_embedding = np.array(question_embedding).reshape(1, -1)  # Reshape for cosine similarity
    except Exception as e:
        print(f"Error generating embedding for the question: {e}")
        return ""

    # Initialize variables to track the best match
    best_score = 0
    best_answer = ""

    # Compare against precomputed embeddings in stored_data
    for entry in custom_model.stored_data:
        try:
            stored_question_embedding = np.array(entry["embeddings"]["question_embedding"]).reshape(1, -1)
            stored_answer = entry["inputs"]["answer"]

            # Compute cosine similarity between the new question and stored question embeddings
            score = cosine_similarity(question_embedding, stored_question_embedding)[0][0]
            print(f"Similarity score: {score}")

            # Update the best match if score is higher than the current best and exceeds the threshold
            if score > best_score and score > 0.8:  # 0.8 is a configurable threshold
                best_score = score
                best_answer = stored_answer
                print(f"New best match found with score {score}: {best_answer}")
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            continue

    # Return the best answer found or an empty string if no match
    if best_answer:
        print(f"bert_processor.py:findAnswerFromBertModel:Returning answer: {best_answer}")
    else:
        print("bert_processor.py:findAnswerFromBertModel:No suitable match found.")
    return best_answer

#
#
#
def findAnswerForQuestion(question):
    """
    Searches for the best matching question in the saved BERT entailments and returns the associated answer.
    If no match is found, returns an empty string.

    Parameters:
        question (str): The input question provided by the user.

    Returns:
        str: The answer associated with the matched question, or an empty string if no match is found.
    """
    print(f"bert_processor.py:findAnswerForQuestion:function called!")
    # lets see how the model is learning
    model_answer = findAnswerFromBertModel(question)
    
    
    
    # Initialize variables to track the best match
    best_question_score = 0
    best_answer = ""

    # Iterate through the saved entailments
    for entry in bert_entailments:
        candidate_question = entry["inputs"]["question"]  # Retrieve the stored question
        inputs = tokenizer(question, candidate_question, return_tensors="pt", truncation=True, padding=True)

        # Perform inference with the custom model
        with torch.no_grad():
            logits = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probabilities = torch.softmax(logits, dim=1).detach().numpy()

        # Extract the entailment probability and label
        entailment_prob = probabilities[0][2]  # Probability of entailment
        if entailment_prob > best_question_score and entailment_prob > 0.8:  # Check if it's the best match and above threshold
            print(f"bert_processor.py:findAnswerForQuestion:Found better match with score: {entailment_prob}")
            best_question_score = entailment_prob
            best_answer = entry["inputs"]["answer"]  # Retrieve the associated answer

    # Return the best answer found, or an empty string if no match
    if best_answer:
        print(f"bert_processor.py:findAnswerForQuestion:Returning answer: {best_answer}")
    else:
        print(f"bert_processor.py:findAnswerForQuestion:No match found.")
        
    print(f"bert_processor.py:findAnswerForQuestion:Bert Model Answer: {model_answer}")
    return best_answer

#
#
#
def generateContextResponseFromBert(question, actual_output, golden_response=None):
    """
    Finds the best matching question in BERT entailments and compares the retrieved answer
    with the golden response if provided.

    Parameters:
        question (str): The input question provided by the user.
        actual_output (str): The actual output (unused here but retained for potential future use).
        golden_response (str, optional): The gold standard response for comparison.

    Returns:
        str: The best matching answer if found, or a message indicating no suitable match.
    """
    print("bert_processing.py:generateContextResponseFromBert:function called!")

    # Step 1: Find the best matching answer for the given question
    # best_answer = findAnswerForQuestion(question)
    best_answer = findAnswerFromBertModel(question)

    # Step 2: If no match is found, return a default response
    if not best_answer:
        print("No suitable question match found in BERT entailments.")
        return "No suitable question match found in BERT entailments."

    # Step 3: If no golden response is provided, return the best answer directly
    if not golden_response:
        print("No golden response provided. Returning the best answer.")
        return best_answer

    # Step 4: Compare the best answer with the golden response
    print("Comparing the answer with the golden response...")
    try:
        # Generate embeddings for both the best answer and the golden response
        answer_embedding = bert_generate_embedding(best_answer)
        golden_embedding = bert_generate_embedding(golden_response)

        # Ensure embeddings are valid
        if answer_embedding is None or golden_embedding is None:
            print("Error generating embeddings for answer or golden response.")
            return "Error generating embeddings for comparison."

        # Compute cosine similarity between embeddings
        similarity_score = cosine_similarity(answer_embedding, golden_embedding)[0][0]
        print(f"Similarity score between answer and golden response: {similarity_score:.2f}")

        # Determine entailment label and return the best answer or a message
        if similarity_score > 0.8:
            return best_answer
        elif similarity_score > 0.5:
            return best_answer
        else:
            return "No suitable match between golden response and answer."

    except Exception as e:
        print(f"Error in comparison: {e}")
        return "Error occurred during comparison with golden response."

#
#
#
def bert_generate_embedding(text):
    """
    Generate embeddings for a given text using the BERT model.

    Parameters:
        text (str): The input text for which embeddings are to be generated.

    Returns:
        np.ndarray: The generated embeddings as a numpy array.
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Generate embeddings using the custom model's BERT component
        with torch.no_grad():
            embeddings = custom_model.bert(**inputs).last_hidden_state[:, 0, :].numpy()  # CLS token
        return embeddings

    except Exception as e:
        print(f"Error in bert_generate_embedding: {e}")
        return None

# Ensure you download NLTK resources if not already done
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocesses input text by normalizing, removing unnecessary characters,
    reducing spaces, and applying lemmatization.
    
    Parameters:
        text (str): The input text to preprocess.
        
    Returns:
        str: The preprocessed text.
    """
    print(f"bert_processing.py:preprocess_text:text: {text}")
    # Convert to lowercase
    text = text.lower()
    
    # Reduce multiple spaces to a single space
    text = " ".join(text.split())
    
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKD", text)
    
    # Remove special characters but keep punctuation like .,!?'" and alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9.,!?'\"]+", " ", text)
    
    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens to their root form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Rejoin tokens into a preprocessed string
    return " ".join(tokens)
#
#
#
def analyze_entailment_with_bert(question, answer, provided_response):
    """
    Perform entailment analysis and similarity evaluation using a BERT model.

    Parameters:
        question (str): The input question.
        answer (str): The candidate answer.
        provided_response (str, optional): A reference or gold-standard response for comparison.

    Returns:
        dict: A dictionary with entailment and similarity analysis results:
            - entailment_label (str): Entailment level ('match', 'partial_match', 'no_match').
            - entailment_prob (float): Probability of entailment for question and answer.
            - neutral_prob (float): Probability of neutrality for question and answer.
            - contradiction_prob (float): Probability of contradiction for question and answer.
            - similarity_score (float): Cosine similarity between question and answer embeddings.
            - provided_response_score (float, optional): Probability of entailment for provided response.
            - similarity_score_provided (float, optional): Cosine similarity between answer and provided response embeddings.
    """
    print(f"bert_processing.py:analyze_entailment_with_bert:Question: {question}")
    print(f"bert_processing.py:analyze_entailment_with_bert:Answer: {answer}")
    print(f"bert_processing.py:analyze_entailment_with_bert:Gold Standard Answer: {provided_response}")
    
    try:
        #Step A: sanitize the question and answer
        question = preprocess_text(question)
        answer = preprocess_text(answer)
        provided_response = preprocess_text(provided_response)
       
        
        # Step 1: Tokenize and encode the question and answer
        inputs = tokenizer(question, answer, return_tensors="pt", truncation=True, padding=True)
        outputs = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs

        # Step 2: Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).detach().numpy()
        entailment_prob = probabilities[0][2]  # Probability of entailment
        neutral_prob = probabilities[0][1]     # Probability of neutrality
        contradiction_prob = probabilities[0][0]  # Probability of contradiction

        # Step 3: Determine the entailment label
        if entailment_prob > 0.8:
            entailment_label = "match"
        elif neutral_prob > 0.5:
            entailment_label = "partial_match"
        else:
            entailment_label = "no_match"

        # Step 4: Generate embeddings for question and answer
        question_embedding = bert_generate_embedding(question)
        answer_embedding = bert_generate_embedding(answer)
        similarity_score = cosine_similarity(question_embedding, answer_embedding)

        # Step 5: Process the provided response if applicable
        provided_response_score = None
        similarity_score_provided = None
        provided_entailment_label = None
        
        if provided_response:
            print(f"bert_processing.py:analyze_entailment_with_bert:provided_response:True")
            inputs_provided = tokenizer(answer, provided_response, return_tensors="pt", truncation=True, padding=True)
            outputs_provided = custom_model(input_ids=inputs_provided['input_ids'], attention_mask=inputs_provided['attention_mask'])
            probabilities_provided = torch.softmax(outputs_provided, dim=1).detach().numpy()
            provided_response_score = probabilities_provided[0][2]
            provided_response_embedding = bert_generate_embedding(provided_response)
            similarity_score_provided = cosine_similarity(answer_embedding, provided_response_embedding)
            
            print("Answer Tokens:", tokenizer.tokenize(answer))
            print("Provided Response Tokens:", tokenizer.tokenize(provided_response))
            
            # Determine the entailment label
            if provided_response_score > 0.6:
                provided_entailment_label = "match"
            elif probabilities_provided[0][1] > 0.3:
                provided_entailment_label = "partial_match"
            else:
                provided_entailment_label = "no_match"
            
            
            
        else:
            print(f"bert_processing.py:analyze_entailment_with_bert:provided_response:Not provided")
            
            
        # Step 6: Create the entailment to train bert
        save_entailment_data(
            question=question,
            answer=answer,
            entailment_label=entailment_label,
            contradiction_prob=contradiction_prob,
            neutral_prob=neutral_prob,
            entailment_prob=entailment_prob,
            provided_response=provided_response,
            provided_response_score=provided_response_score
        )
        
        # Step 7: Construct the result dictionary
        result = {
            "question": question,  # Input question
            "answer": answer,  # Generated answer
            "provided_response": provided_response,  # Provided (gold standard) response
            "entailment_label": entailment_label,  # Question vs. Answer comparison
            "contradiction_prob": float(contradiction_prob),  # Question vs. Answer
            "neutral_prob": float(neutral_prob),  # Question vs. Answer
            "entailment_prob": float(entailment_prob),  # Question vs. Answer
            "provided_response_score": float(provided_response_score) if provided_response else None,  # Answer vs. Gold Standard
            "provided_entailment_label": provided_entailment_label  # Answer vs. Gold Standard
        }

        return result

    except Exception as e:
        print(f"Error in analyze_entailment_with_bert: {e}")
        return {
            "entailment_label": "no_match",
            "entailment_prob": 0.0,
            "neutral_prob": 0.0,
            "contradiction_prob": 0.0,
            "similarity_score": 0.0,
            "provided_response_score": None,
            "similarity_score_provided": None
        }
#
#
#
def save_entailment_atomic(entailment_path, entailment_record):
    """
    Save entailment data to a JSON file using atomic file writing.

    Parameters:
        entailment_path (str): The path to the JSON file where entailment data is stored.
        entailment_record (dict): The new entailment record to be added.
    """
    print(f"bert_processing.py:save_entailment_atomic:provided_response:Function Called")
    try:
        # Load existing data if the file exists
        if os.path.exists(entailment_path):
            with open(entailment_path, "r") as file:
                try:
                    data = json.load(file)  # Load existing data
                except json.JSONDecodeError:
                    print(f"{entailment_path} is empty or corrupted. Initializing with an empty list.")
                    data = []
        else:
            data = []  # Initialize with an empty list if the file does not exist

        # Append the new entailment record
        data.append(entailment_record)

        # Write to a temporary file
        temp_path = f"{entailment_path}.tmp"
        with open(temp_path, "w") as temp_file:
            json.dump(data, temp_file, indent=4)

        # Replace the original file with the temporary file
        os.replace(temp_path, entailment_path)
        print(f"bert_processor.py:save_entailment_data:Entailment saved: {entailment_record}")

    except Exception as e:
        print(f"bert_processor.py:save_entailment_data:Error saving entailment data: {e}")
#
#
#
def save_entailment_data(question, answer, entailment_label, contradiction_prob, neutral_prob,
                         entailment_prob, provided_response=None, provided_response_score=None):
    """
    Save entailment data and update the BERT model.
    """
    global entailment_path
    global custom_model
    global model_path

    print("bert_processor.py:save_entailment_data: Function called")
    
     # Generate embeddings for the question, answer, and provided response
    try:
        question_embedding = bert_generate_embedding(question).tolist()  # Convert to list for JSON compatibility
        answer_embedding = bert_generate_embedding(answer).tolist()
        provided_response_embedding = (
            bert_generate_embedding(provided_response).tolist() if provided_response else None
        )
    except Exception as e:
        print(f"bert_processor.py:save_entailment_data:Error generating embeddings: {e}")
        return
    
    # Create a dictionary to save
    entailment_record = {
        "inputs": {
            "question": question,
            "answer": answer,
            "provided_response": provided_response  # Add provided response to inputs
        },
        "embeddings": {
            "question_embedding": question_embedding,
            "answer_embedding": answer_embedding,
            "provided_response_embedding": provided_response_embedding
        },
        "entailment_label": entailment_label,
        "contradiction_prob": float(contradiction_prob),
        "neutral_prob": float(neutral_prob),
        "entailment_prob": float(entailment_prob),
        "provided_response_score": float(provided_response_score) if provided_response_score else None
    }
    # Step 1: Add to the custom_model's stored data
    if not hasattr(custom_model, "stored_data"):
        print("bert_processor.py:save_entailment_data:Initializing custom_model.stored_data as a list.")
        custom_model.stored_data = []  # Initialize if not already present
    
    # save the entailment data to the stored_data
    custom_model.stored_data.append(entailment_record)
    print(f"bert_processor.py:save_entailment_data:Entailment record added. Total records: {len(custom_model.stored_data)}")
    
    # Step 2: Save the model's updated weights and stored data
    try:
        # Save the model's state and stored data in a single `.pth` file
        torch.save({
            "model_state_dict": custom_model.state_dict(),
            "stored_data": custom_model.stored_data  # Save stored data alongside model weights
        }, model_path)
        print(f"bert_processor.py:save_entailment_data:Model and stored data saved to {model_path}")
    except Exception as e:
        print(f"bert_processor.py:save_entailment_data:Error saving model state and data: {e}")

    # Step 3: Save the entailment record to json file using Atomic file write
    save_entailment_atomic(entailment_path, entailment_record)

    # Step 4: Reload the entailments into memory
    try:
        global bert_entailments
        bert_entailments = load_saved_entailments()  # Reload updated JSON into memory
        print(f"bert_processor.py:save_entailment_data:Reloaded {len(bert_entailments)} entailments into memory.")
    except Exception as e:
        print(f"bert_processor.py:save_entailment_data:Error reloading entailments: {e}")

#
#
#
def get_training_info():
    """
    Gathers training information from bert_entailments.
    Returns the total number of trained items, questions, and answers.
    """
    print(f"bert_processor.py:get_training_info:Function called")
    try:
        global bert_entailments  # Use the globally loaded entailments

        if not bert_entailments:  # If no entailments are loaded
            print(f"bert_processor.py:get_training_info:bert_entailments:has no entries")
            return {
                "total_trained_items": 0,
                "questions_trained": 0,
                "answers_trained": 0
            }

        # Count the number of questions and answers trained
        print(f"bert_processor.py:get_training_info:bert_entailments:populate the data")
        questions_trained = len([entry["inputs"]["question"] for entry in bert_entailments if "question" in entry["inputs"]])
        answers_trained = len([entry["inputs"]["answer"] for entry in bert_entailments if "answer" in entry["inputs"]])
        total_trained_items = len(bert_entailments)  # Total entries
    
        return {
            "total_trained_items": total_trained_items,
            "questions_trained": questions_trained,
            "answers_trained": answers_trained
        }

    except Exception as e:
        print(f"bert_processor.py:get_training_info:Error: {e}")
        return {
            "error": str(e)
        }

