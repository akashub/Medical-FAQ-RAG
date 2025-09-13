# 🏥 Medical FAQ Chatbot

This project is a sophisticated Medical FAQ Chatbot built using a Retrieval-Augmented Generation (RAG) architecture. It leverages a hybrid search mechanism to provide accurate, context-aware answers to medical questions based on a large knowledge base. The application is built with Python and features an interactive user interface powered by Streamlit.



## ✨ Features

* **Interactive Chat Interface:** A user-friendly, web-based chat interface built with Streamlit.
* **Hybrid Search Retrieval:** Combines the strengths of sparse (keyword-based) and dense (semantic) search for superior accuracy.
    * **BM25:** For efficient, keyword-based retrieval (sparse).
    * **ChromaDB:** For semantic similarity search using dense vector embeddings.
* **Reciprocal Rank Fusion (RRF):** Intelligently merges results from both search methods to provide the most relevant documents.
* **Advanced AI Generation:** Uses Google's powerful **Gemini 2.5 Flash** model to generate clear, concise, and helpful answers based on the retrieved information.
* **Large Knowledge Base:** Powered by the [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset), containing over 43,000 question-answer pairs.
* **Persistent Vector Store:** ChromaDB is used to store document embeddings, ensuring that data is only processed once, leading to fast subsequent startups.

***

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline to answer user queries.

1.  **User Query:** A user asks a medical question through the Streamlit interface.
2.  **Hybrid Search:** The query is sent to two parallel retrieval systems:
    * The **BM25 retriever** performs a fast keyword search.
    * **ChromaDB** performs a semantic search to find conceptually similar documents.
3.  **Result Fusion (RRF):** The ranked lists of documents from both retrievers are combined using Reciprocal Rank Fusion to produce a single, highly relevant list of context documents.
4.  **Prompt Augmentation:** The user's original query and the content of the top-ranked documents are formatted into a detailed prompt.
5.  **Answer Generation:** The augmented prompt is sent to the **Gemini 2.5 Flash** model, which generates a final, human-readable answer.

***

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Web Framework:** Streamlit
* **AI/LLM:** Google Gemini 2.5 Flash
* **Vector Database:** ChromaDB
* **Search Algorithms:** BM25 (via `rank_bm25`), Dense Vector Search
* **Core Libraries:** `google-generativeai`, `chromadb`, `rank_bm25`, `streamlit`

***

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.9 or newer
* A Google Gemini API Key. You can get one for free from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 4. Install Dependencies
Create a ```requirements.txt``` file with the following content:

```plaintext
streamlit
python-dotenv
google-generativeai
chromadb
rank_bm25
```

---


Then, install all the required packages:

```bash
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables
Create a file named ```.env``` in the root of your project directory and add your Google Gemini API key:

```python
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

---

### 6. Download the Dataset

- Go to the Kaggle dataset page.

- Download the dataset. You will likely get a file named Medical_Q&A.json.

- Create a data folder in your project's root directory.

- Place the downloaded JSON file in the data folder and rename it to medical_faqs.json.

---

### 7. Run the Application
Execute the following command in your terminal:

```Bash

streamlit run app.py
```

The application will open in your web browser. The first run will be slow as it processes the 43,000+ documents and builds the vector database. Subsequent runs will be much faster.

---

## 📜 Disclaimer
⚠️ This chatbot provides general medical information for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.


