# Medical Assistant AI using RAG
<p align="center">
  <img src="RAG overview.png" alt="How RAG works">
</p>

## Overview
This notebook develops a Medical Assistant AI solution using a Retrieval Augmented Generation (RAG) framework. It leverages a Large Language Model (Mistral-7B) combined with the Merck Manuals (a comprehensive medical reference) to provide accurate and context-aware answers to clinical questions. The solution aims to address the challenges faced by healthcare professionals in managing vast volumes of medical data and making timely diagnostic decisions.

## Business Problem
The healthcare industry grapples with information overload, making it challenging for professionals to access and synthesize critical medical knowledge efficiently. This leads to difficulties in delivering accurate diagnoses, formulating timely treatment plans, and maintaining high standards of care, especially in time-sensitive situations. The objective is to create an AI solution that streamlines access to reliable medical information, supports quick decision-making, and enhances overall operational effectiveness in healthcare settings.

## Data
The primary data source for this project is the **Merck Manuals**, a renowned medical reference. The manual is provided as a PDF document, consisting of over 4,000 pages divided into 23 sections. It covers a wide range of medical topics, including disorders, tests, diagnoses, and drugs. This comprehensive manual serves as the knowledge base for the RAG system.

## Approach
The solution involves building a RAG-based AI system. The key steps include:
1.  **Installing Libraries**: Essential Python libraries like `llama-cpp-python`, `langchain`, `pymupdf`, `chromadb`, and `sentence-transformers` are installed.
2.  **LLM Setup**: A `Mistral-7B-Instruct-v0.2-GGUF` model is downloaded and loaded using `llama-cpp-python` for initial question answering.
3.  **Data Preparation**: The Merck Manuals PDF is loaded using `PyMuPDFLoader`, chunked into smaller documents using `RecursiveCharacterTextSplitter`, and then embedded using `SentenceTransformerEmbeddings` (`thenlper/gte-large`).
4.  **Vector Database**: The embedded document chunks are stored in a `Chroma` vector database for efficient retrieval.
5.  **Retriever Setup**: A retriever is configured to fetch the most relevant document chunks based on a user query.
6.  **Prompt Engineering**: A system prompt and a user message template are designed to guide the LLM in providing precise, evidence-based responses from the retrieved context.
7.  **RAG-based Question Answering**: The system combines the user query with retrieved context from the vector database and the engineered prompts to generate informed medical answers.
8.  **Fine-tuning**: Various parameters like `temperature`, `max_tokens`, `top_p`, `top_k`, and `k` (for retriever) are adjusted to optimize the response quality.
9.  **Output Evaluation**: The RAG system's performance is evaluated using an LLM-as-a-judge approach, assessing 'groundedness' (adherence to context) and 'relevance' (alignment with the question).

## Results
The implementation of the RAG system demonstrated significant improvements in the quality and accuracy of medical responses compared to a standalone LLM. The evaluation using the LLM-as-a-judge method showed:

*   **High Groundedness (Score 5)**: The model consistently scored a 5 for groundedness, indicating that the generated answers were derived *only* from the information presented in the provided context, preventing hallucination.
*   **Good Relevance (Score 4)**: The model generally scored a 4 for relevance, meaning the answers effectively addressed the main aspects of the questions based on the context, covering important details without including irrelevant information.

These results confirm the feasibility and effectiveness of using a RAG approach with specialized medical manuals to provide reliable clinical information.

## Tools & Technologies
*   **Python**: Programming language.
*   **llama-cpp-python**: For loading and running the GGUF formatted LLM locally.
*   **Hugging Face Hub**: For downloading the Mistral-7B-Instruct LLM.
*   **Langchain**: Framework for developing LLM applications, used for text splitting, document loading, and integration with vector stores and embeddings.
*   **PyMuPDFLoader**: Langchain document loader for processing PDF files.
*   **tiktoken**: Tokenizer used for character text splitting.
*   **SentenceTransformerEmbeddings**: For generating vector embeddings of document chunks using `thenlper/gte-large` model.
*   **Chroma**: An open-source vector database for storing and retrieving document embeddings.
*   **pandas**: For data manipulation and analysis.
*   **Mistral-7B-Instruct-v0.2-GGUF**: The Large Language Model used for generating responses.

## Key Learnings
1.  **Contextual Relevance**: Increasing `chunk_overlap` in the retriever can enhance result relevance, especially for sequential medical instructions.
2.  **Groundedness**: Strict prompt engineering is crucial for achieving high groundedness, ensuring responses are solely based on provided context.
3.  **Domain-Specific Embeddings**: Future improvements could involve using embedding models pre-trained on medical datasets for even better document retrieval accuracy.
4.  **Continuous Knowledge Update**: Regular updates to the knowledge base with the latest medical research are essential for maintaining relevance and accuracy.
5.  **Scalability**: The RAG system can be expanded to support additional medical specialties, broadening its utility.
