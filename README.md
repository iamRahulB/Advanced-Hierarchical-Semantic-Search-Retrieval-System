# Advanced-Hierarchical-Semantic-Search-Retrieval-System
This project implements an advanced hierarchical semantic search retrieval system designed to efficiently extract relevant information from documents. Utilizing state-of-the-art clustering and semantic search algorithms, the system can accurately identify and retrieve specific data points,



## Plan for Building a Retrieval-Augmented Generation (RAG) System

### 1. Text Clustering with Hierarchical Clustering
- **Objective**: Group similar text chunks into clusters.
- **Method**: Use hierarchical clustering to organize the text data into meaningful clusters.

### 2. Text Embeddings and Semantic Search
- **Objective**: Convert text chunks into embeddings for efficient semantic search.
- **Method**:
  - Pass all text chunks through an embedding model to generate vector representations.
  - Perform a semantic search on these embeddings to find the most relevant text chunks.

### 3. Combining Semantic Search with Clustering
- **Objective**: Enhance the retrieval process by leveraging both semantic search and clustering.
- **Method**:
  1. **Semantic Search**:
     - Perform a semantic search on the embeddings.
     - Retrieve the top 2 most relevant text chunks based on the search results.
  2. **Cluster Identification**:
     - Identify the clusters to which these top 2 text chunks belong.
  3. **Nearest Text Chunk Retrieval**:
     - Within each identified cluster, find the 5 nearest text chunks to the top search results.
     - This ensures that the retrieved text chunks are not only semantically relevant but also contextually similar within their clusters.

### 4. Expected Outcome
- By combining semantic search with hierarchical clustering, the system will retrieve text chunks that are both semantically and contextually relevant, improving the overall quality and relevance of the results.
