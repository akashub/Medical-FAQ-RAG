# 🏥 Medical FAQ Chatbot

This project is a sophisticated Medical FAQ Chatbot built using a Retrieval-Augmented Generation (RAG) architecture. It leverages a hybrid search mechanism to provide accurate, context-aware answers to medical questions based on a large knowledge base. The application is built with Python and features an interactive user interface powered by Streamlit.

---

## Who Am I?
Hey! I am Aakash Singh. I am a recent BTech CSE graduate specializing in Data Science and AI Research, with expertise in AI Copilots, RAG systems, Deep Learning, production ML systems, and LLMs. 

I completed my semester abroad at University of Florida (GPA: 3.85/4) where I pursued post-grad level courses eventually gaining the prestigious **Achievement Award Scholarship** worth **$4500** with an admit to the prestigious MS in CS at UF. 

I have a proven track record delivering AI solutions including an AI Copilot that boosted SOC efficiency by 60%, research deep learning models,
and scalable RAG pipelines. Combines research foundation with industry experience deploying data-driven solutions.



## ✨ Features

* **Interactive Chat Interface:** A user-friendly, web-based chat interface built with Streamlit.
* **Hybrid Search Retrieval:** Combines the strengths of sparse (keyword-based) and dense (semantic) search for superior accuracy.
    * **BM25:** For efficient, keyword-based retrieval (sparse).
    * **ChromaDB:** For semantic similarity search using dense vector embeddings.
* **Reciprocal Rank Fusion (RRF):** Intelligently merges results from both search methods to provide the most relevant documents.
* **Advanced AI Generation:** Uses Google's powerful **Gemini 2.5 Flash** model to generate clear, concise, and helpful answers based on the retrieved information.
* **Large Knowledge Base:** Powered by the [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset), containing over 43,000 question-answer pairs.
* **Persistent Vector Store:** ChromaDB is used to store document embeddings, ensuring that data is only processed once, leading to fast subsequent startups.

  <img width="1851" height="956" alt="image" src="https://github.com/user-attachments/assets/cc67c8b1-2611-4206-af95-bd15e187ae4e" />

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

   <img width="1851" height="956" alt="image" src="https://github.com/user-attachments/assets/967e8a23-6b97-468e-a53f-2ad63257e353" />

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

## 🚀 Potential Pipeline Enhancements

The current implementation uses a solid hybrid search foundation (BM25 + Dense Retrieval). To further enhance its accuracy, robustness, and contextual understanding, the pipeline can be expanded with the following advanced techniques.

### 1. Query Transformation Techniques

The quality of retrieval is highly dependent on the quality of the user's query. These techniques refine the initial query to improve search results.

* **Multi-Query Retriever:** Instead of using a single query, this approach uses an LLM to generate several variations of the user's question from different perspectives. For example, if the user asks, "What are the side effects of paracetamol?", the LLM might generate:
    1.  "Common adverse reactions to paracetamol"
    2.  "Long-term effects of taking acetaminophen"
    3.  "Paracetamol risks and warnings"
    The pipeline would then retrieve documents for all three queries, creating a richer, more diverse set of results.

* **HyDE (Hypothetical Document Embeddings):** This is a powerful zero-shot technique. Instead of embedding the user's (often short) query, we first ask the LLM to generate a detailed, hypothetical answer to the question. Then, we create an embedding of this *hypothetical document* and use it for the vector search. This often yields more relevant results because the embedding of a full-text document is more likely to be semantically similar to the actual documents in the knowledge base.

### 2. Advanced Fusion and Re-Ranking

Once we have documents from multiple sources (BM25, Dense Search, or multiple queries), we need to intelligently rank them.

* **RAG-Fusion:** This technique builds on the Multi-Query approach. After retrieving documents for each generated query variant, all the results are collected. A re-ranking algorithm, like **Reciprocal Rank Fusion (RRF)**, is then applied to the combined set. This method scores documents based on their rank across the different result lists, pushing the most consistently high-ranked documents to the top. This ensures the final context sent to the LLM is the most relevant and comprehensive.

### 3. Agentic Workflow with Self-Correction

This transforms the linear pipeline into an intelligent agent that can reason about its own results and self-correct, often implemented with a framework like **LangGraph**.

* **The Retrieve-Grade-Correct Loop:**
    1.  **Retrieve:** Fetch documents using one of the methods above (e.g., RAG-Fusion).
    2.  **Grade:** A new step is introduced where an LLM acts as a "grader." It examines the retrieved documents and answers a simple question: "Is the information in these documents relevant to the user's original question?"
    3.  **Conditional Logic (The "Agent" part):**
        * **If Relevant:** The documents are passed to the final LLM to generate the answer for the user.
        * **If Not Relevant:** The agent decides the retrieval failed. It can then trigger a corrective action, such as using a different query transformation technique (e.g., applying HyDE if it started with Multi-Query) and **looping back to the retrieve step**. This cycle continues until relevant documents are found or a maximum number of attempts is reached.

This self-correcting loop makes the system incredibly robust and resilient to ambiguous user questions or initial retrieval failures.

---

## 📜 Disclaimer
⚠️ This chatbot provides general medical information for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.


