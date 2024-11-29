# Text Classifier Approximate Cache (TCAC)

## Introduction
Performing text classification has become really easy in these days of Large Language Models. All, you need to do is ask an LLM to classify the document into the various classes.

But there is a catch. If you have to classify millions of documents, suddenly the cost of classification would become pretty heavy. 

How about building your own classifier based on embedding? 
But wait, who will label the documents Initially? 
What if we could use the LLM such as ChatGPT or Gemini to label the documents. 

That's exactly what TCAC is all about. In the begining, when you ask TCAC to classify a document, it would send the document to an LLM such as ChatGPT or Gemini to classify and then send the result to the end user but it keep the a copy of the document and label.

### Labeling Phase

    Step 1. User --- (Document) --> TCAC --("Is this document related to billing?", Document) ---> ChatGPT/Gemini

    Step 2.a User <---- (Document, Label: Yes/No) <--- TCAC <--- (Document, Label: Yes/No) --- ChatGPT/Gemini

    Step 2.b TCAC --- (Document, Label) --> Database

Once it has gathered minimum dataset, it would split the data set into training and test. And start to training it's model.

### Training Phase
    Training Data (Document + Labels) --- (Training / Model.FIT / KNN) --> Model

Next step would be to validate if the model's performance on test dataset is good enough. If it is above the predefined thresold, it would deploy this model. The user's document would not longer be going to the LLM.

### Inference Phase

    Step 1. User --- (Document) --> TCAC -->( Internal Model)
    Step 2. User <---- (Document, Label: Yes/No) <--- TCAC 

This would eventually reduce the cost of document classification to zero.

TCAC is a one of it's kind product which is basically behaves like an approximate cache unlike the usual cache which does an exact look up.

## API Signature

```
classifier_space = TCAC.create_classifier_space("org.financialdocs")

invoiceClassifer = classifier_space.binaryClassifer(name="invoice", prompt="Please classify the following document into Invoice or Not.")
result = invoiceClassifer.predict("Invoice#123 Purchase of xyz...")

```


