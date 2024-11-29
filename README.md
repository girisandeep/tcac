# Text Classifier Approximate Cache (TCAC)

## Overview

Text classification has become incredibly accessible with the advent of Large Language Models (LLMs). By simply querying an LLM, you can classify documents into various categories. 

However, there's a challenge: classifying millions of documents can quickly become prohibitively expensive. 

What if you could create your own classifier based on embeddings? And what if LLMs like ChatGPT or Gemini could label your documents initially? 

This is where **TCAC** comes in. Initially, TCAC leverages an LLM to classify documents and stores the document-label pairs. Over time, it trains its own internal model to classify documents without relying on the LLM, drastically reducing classification costs.

---

## How TCAC Works

### 1. **Labeling Phase**
During the labeling phase, TCAC relies on LLMs to label documents and stores the results for future use.

1. User submits a document for classification to TCAC.  
   TCAC forwards the query to an LLM (e.g., "Is this document related to billing?").
2. LLM returns the classification label to TCAC.  
   TCAC provides the result to the user and saves the document-label pair in its database.

**Workflow Example:**  
```
Step 1: User ---> (Document) ---> TCAC ---> (Prompt + Document) ---> LLM (ChatGPT/Gemini)  
Step 2: LLM ---> (Document, Label: Yes/No) ---> TCAC ---> User  
Step 3: TCAC ---> (Document, Label) ---> Database  
```

---

### 2. **Training Phase**
Once TCAC has collected enough labeled data, it splits the data into training and test sets. Using this data, it trains a model (e.g., using KNN or other algorithms).

- If the model’s performance on the test data exceeds a predefined threshold, TCAC deploys the model for classification.

---

### 3. **Inference Phase**
Once the model is deployed, TCAC no longer queries the LLM for predictions. Instead, it uses its internal model for document classification, significantly reducing costs.

**Workflow Example:**  
```
Step 1: User ---> (Document) ---> TCAC ---> Internal Model  
Step 2: TCAC ---> (Document, Label: Yes/No) ---> User  
```

By eliminating reliance on LLMs for inference, TCAC brings classification costs close to zero.

---

## Key Features

- **Adaptive Caching**: TCAC functions as an approximate cache, storing document-label pairs instead of performing exact lookups.  
- **Cost Efficiency**: Transitioning from LLM-based labeling to an internal model drastically reduces long-term costs.  
- **Scalability**: Ideal for large-scale document classification tasks.

---

## API Example

Using TCAC is simple and intuitive:

```python
# Create a classifier space
classifier_space = TCAC.create_classifier_space("org.financialdocs")

# Define a binary classifier
invoice_classifier = classifier_space.binary_classifier(
    name="invoice",
    prompt="Please classify the following document into Invoice or Not."
)

# Use the classifier to make predictions
result = invoice_classifier.predict("Invoice#123 Purchase of xyz...")
```

---

TCAC is a unique solution designed to balance the flexibility of LLMs with the efficiency of traditional machine learning models, making it ideal for organizations looking to optimize large-scale text classification.