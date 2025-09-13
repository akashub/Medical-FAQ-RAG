# import json
# import os
# import chromadb
# import google.generativeai as genai
# from typing import List, Dict
# import logging
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MedicalRAGPipeline:
#     def __init__(self, gemini_api_key: str):
#         """Initialize the RAG pipeline with Gemini API"""
        
#         # Configure Gemini
#         genai.configure(api_key=gemini_api_key)
#         self.llm = genai.GenerativeModel('gemini-pro')
        
#         # Initialize TF-IDF vectorizer for embeddings
#         self.vectorizer = TfidfVectorizer(
#             max_features=1000,
#             stop_words='english',
#             ngram_range=(1, 2)
#         )
        
#         # Store documents and their embeddings
#         self.documents = []
#         self.document_embeddings = None
#         self.metadata = []
        
#         # Initialize ChromaDB (we'll use it for metadata storage)
#         self.chroma_client = chromadb.PersistentClient(path="./vector_store")
#         self.collection_name = "medical_faqs"
        
#         try:
#             self.collection = self.chroma_client.get_collection(name=self.collection_name)
#             logger.info("Loaded existing vector database")
#         except:
#             self.collection = self.chroma_client.create_collection(name=self.collection_name)
#             logger.info("Created new vector database")
    
#     def preprocess_text(self, text: str) -> str:
#         """Basic text preprocessing"""
#         # Convert to lowercase
#         text = text.lower()
#         # Remove extra whitespace
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text
    
#     def load_and_process_data(self, data_path: str):
#         """Load medical FAQ data and create embeddings"""
#         try:
#             with open(data_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
            
#             # Check if we already have data loaded
#             if len(self.documents) > 0:
#                 logger.info("Data already loaded")
#                 return
            
#             documents = []
#             metadata = []
            
#             for i, item in enumerate(data):
#                 # Combine question and answer for better context
#                 doc_text = f"Question: {item['question']} Answer: {item['answer']}"
#                 processed_text = self.preprocess_text(doc_text)
                
#                 documents.append(processed_text)
#                 metadata.append({
#                     "question": item['question'],
#                     "answer": item['answer'],
#                     "doc_id": i
#                 })
            
#             # Create TF-IDF embeddings
#             logger.info("Generating TF-IDF embeddings...")
#             self.document_embeddings = self.vectorizer.fit_transform(documents)
#             self.documents = documents
#             self.metadata = metadata
            
#             # Store in ChromaDB for persistence (optional)
#             try:
#                 # Clear existing data
#                 existing_data = self.collection.get()
#                 if existing_data['ids']:
#                     self.collection.delete(ids=existing_data['ids'])
                
#                 # Add new data
#                 self.collection.add(
#                     documents=documents,
#                     metadatas=metadata,
#                     ids=[f"doc_{i}" for i in range(len(documents))]
#                 )
#                 logger.info(f"Successfully processed {len(documents)} documents")
#             except Exception as e:
#                 logger.warning(f"ChromaDB storage failed: {e}, continuing with in-memory storage")
            
#         except Exception as e:
#             logger.error(f"Error loading data: {str(e)}")
#             raise
    
#     def retrieve_relevant_docs(self, query: str, n_results: int = 3) -> List[Dict]:
#         """Retrieve relevant documents based on query using TF-IDF similarity"""
#         try:
#             if self.document_embeddings is None:
#                 logger.error("No documents loaded")
#                 return []
            
#             # Preprocess query
#             processed_query = self.preprocess_text(query)
            
#             # Transform query using the fitted vectorizer
#             query_embedding = self.vectorizer.transform([processed_query])
            
#             # Calculate cosine similarities
#             similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
            
#             # Get top n_results
#             top_indices = similarities.argsort()[-n_results:][::-1]
            
#             retrieved_docs = []
#             for idx in top_indices:
#                 if similarities[idx] > 0:  # Only include docs with some similarity
#                     retrieved_docs.append({
#                         'question': self.metadata[idx]['question'],
#                         'answer': self.metadata[idx]['answer'],
#                         'similarity': float(similarities[idx])
#                     })
            
#             return retrieved_docs
            
#         except Exception as e:
#             logger.error(f"Error retrieving documents: {str(e)}")
#             return []
    
#     def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
#         """Generate response using Gemini API"""
#         try:
#             # Prepare context from retrieved documents
#             if not retrieved_docs:
#                 context = "No relevant medical information found in the knowledge base."
#             else:
#                 context = "Relevant Medical Information from Knowledge Base:\n\n"
#                 for i, doc in enumerate(retrieved_docs, 1):
#                     context += f"{i}. Question: {doc['question']}\n"
#                     context += f"   Answer: {doc['answer']}\n"
#                     context += f"   Relevance: {doc['similarity']:.2f}\n\n"
            
#             # Create prompt
#             prompt = f"""You are a helpful medical information assistant. Based on the provided medical knowledge base, answer the user's question clearly and accurately.

# IMPORTANT GUIDELINES:
# - Always remind users to consult healthcare professionals for personalized medical advice
# - Do not provide specific medical diagnoses
# - If the retrieved information doesn't fully answer the question, acknowledge this
# - Be clear, concise, and helpful
# - Focus on general medical information only

# {context}

# User Question: {query}

# Please provide a helpful response based on the available information:"""

#             # Generate response using Gemini
#             response = self.llm.generate_content(prompt)
#             return response.text
            
#         except Exception as e:
#             logger.error(f"Error generating response: {str(e)}")
#             return "I apologize, but I'm having trouble generating a response right now. Please try again later or consult with a healthcare professional for medical advice."
    
#     def chat(self, query: str) -> str:
#         """Main chat function that combines retrieval and generation"""
#         try:
#             # Retrieve relevant documents
#             retrieved_docs = self.retrieve_relevant_docs(query, n_results=3)
            
#             if not retrieved_docs:
#                 return "I couldn't find relevant information for your query in my knowledge base. Please rephrase your question or consult with a healthcare professional for medical advice."
            
#             # Generate response
#             response = self.generate_response(query, retrieved_docs)
#             return response
            
#         except Exception as e:
#             logger.error(f"Error in chat: {str(e)}")
#             return "I apologize, but I encountered an error. Please try again or consult with a healthcare professional for medical advice."



import json
import os
import chromadb
import google.generativeai as genai
from typing import List, Dict
import logging
import re
from rank_bm25 import BM25Okapi


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalRAGPipeline:
    def __init__(self, gemini_api_key: str):
        """Initialize the RAG pipeline with Gemini API and retrieval models."""
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        
        # --- ChromaDB for Dense Retrieval ---
        self.chroma_client = chromadb.PersistentClient(path="./vector_store")
        self.collection_name = "medical_faqs"
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        logger.info("Initialized ChromaDB connection for dense retrieval.")
        
        # --- BM25 for Sparse Retrieval ---
        self.bm25 = None
        self.bm25_documents = []
        self.bm25_doc_ids = [] # To map BM25 results back to original data
        logger.info("BM25 retriever initialized (will be populated during data loading).")
    

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_and_process_data(self, data_path: str):
        """Load data and prepare it for both dense (ChromaDB) and sparse (BM25) retrieval."""
        try:
            # Check if the collection is already populated.
            if self.collection.count() > 0:
                logger.info("Data already exists in ChromaDB. Re-initializing BM25 from existing data.")
                # We still need to initialize BM25 as it's in-memory
                existing_data = self.collection.get(include=["documents", "metadatas"])
                tokenized_corpus = [doc.split(" ") for doc in existing_data['documents']]
                self.bm25_doc_ids = existing_data['ids']
                self.bm25_documents = [
                    {"id": doc_id, "text": text, "metadata": meta}
                    for doc_id, text, meta in zip(existing_data['ids'], existing_data['documents'], existing_data['metadatas'])
                ]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info("BM25 initialized successfully from existing data.")
                return

            # This block will now only run ONCE, when the database is empty.
            logger.info("No data found in ChromaDB. Starting one-time data processing...")
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            chroma_docs = []
            chroma_metadatas = []
            chroma_ids = []
            tokenized_corpus = []

            for i, item in enumerate(data):
                doc_text = f"Question: {item['question']} Answer: {item['answer']}"
                processed_text = self.preprocess_text(doc_text)
                doc_id = f"doc_{i}"

                chroma_docs.append(processed_text)
                chroma_metadatas.append({"question": item['question'], "answer": item['answer']})
                chroma_ids.append(doc_id)
                tokenized_corpus.append(processed_text.split(" "))
                
            # --- Load data into ChromaDB (ONCE) ---
            logger.info("Adding documents to ChromaDB for dense retrieval...")
            self.collection.add(documents=chroma_docs, metadatas=chroma_metadatas, ids=chroma_ids)
            logger.info(f"Successfully processed {self.collection.count()} documents for ChromaDB.")

            # --- Initialize and "train" BM25 ---
            logger.info("Initializing BM25 with tokenized corpus for sparse retrieval...")
            self.bm25_doc_ids = chroma_ids
            self.bm25_documents = [
                {"id": doc_id, "text": text, "metadata": meta}
                for doc_id, text, meta in zip(chroma_ids, chroma_docs, chroma_metadatas)
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _reciprocal_rank_fusion(self, search_results: List[List[str]], k: int = 60) -> Dict[str, float]:
        """Combines ranked lists of document IDs using RRF."""
        fused_scores = {}
        for doc_list in search_results:
            for rank, doc_id in enumerate(doc_list):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by score in descending order
        reranked_results = {doc_id: score for doc_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)}
        return reranked_results

    def retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve documents using a hybrid search approach (Dense + Sparse)."""
        try:
            processed_query = self.preprocess_text(query)

            # --- 1. Dense Search (ChromaDB) ---
            dense_results = self.collection.query(query_texts=[processed_query], n_results=n_results)
            dense_ids = dense_results['ids'][0]

            # --- 2. Sparse Search (BM25) ---
            tokenized_query = processed_query.split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_results]
            sparse_ids = [self.bm25_doc_ids[i] for i in top_n_indices]

            # --- 3. Fuse Results (RRF) ---
            if not dense_ids and not sparse_ids:
                return []
            
            reranked_ids = self._reciprocal_rank_fusion([dense_ids, sparse_ids])

            # --- 4. Fetch unique documents in the new order ---
            final_ids = list(reranked_ids.keys())[:n_results]
            retrieved_docs_map = {doc['id']: doc['metadata'] for doc in self.bm25_documents}
            
            retrieved_docs = []
            for doc_id in final_ids:
                metadata = retrieved_docs_map.get(doc_id)
                if metadata:
                    retrieved_docs.append({
                        'question': metadata.get('question', ''),
                        'answer': metadata.get('answer', ''),
                        'similarity': reranked_ids.get(doc_id, 0) # Use RRF score as relevance
                    })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
        
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate response using Gemini API."""
        try:
            if not retrieved_docs:
                context = "No relevant medical information found in the knowledge base."
            else:
                context = "Relevant Medical Information from Knowledge Base:\n\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    context += f"{i}. Question: {doc['question']}\n"
                    context += f"   Answer: {doc['answer']}\n"
                    # We use the RRF score as a general relevance indicator
                    context += f"   Relevance Score: {doc['similarity']:.4f}\n\n"
            
            prompt = f"""You are a helpful medical information assistant. Based on the provided medical knowledge base, answer the user's question clearly and accurately.

IMPORTANT GUIDELINES:
- Always remind users to consult healthcare professionals for personalized medical advice
- Do not provide specific medical diagnoses
- If the retrieved information doesn't fully answer the question, acknowledge this
- Be clear, concise, and helpful
- Focus on general medical information only

{context}

User Question: {query}

Please provide a helpful response based on the available information:"""
            response = self.llm.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again later or consult with a healthcare professional for medical advice."

    def chat(self, query: str) -> str:
        """Main chat function that combines retrieval and generation."""
        try:
            retrieved_docs = self.retrieve_relevant_docs(query, n_results=3)
            
            if not retrieved_docs:
                return "I couldn't find relevant information for your query in my knowledge base. Please rephrase your question or consult with a healthcare professional for medical advice."
            
            response = self.generate_response(query, retrieved_docs)
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return "I apologize, but I encountered an error. Please try again or consult with a healthcare professional for medical advice."