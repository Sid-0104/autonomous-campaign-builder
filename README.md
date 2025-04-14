🚀 Autonomous Campaign Builder - Embedding Pipeline
This project provides a pipeline to chunk, embed, and store tabular data from CSVs using either Google Gemini or OpenAI embeddings. It also stores and queries the embeddings using ChromaDB.

📦 Features
 Load all CSVs from a folder structure (1 row = 1 chunk)

 Choose between OpenAI and Gemini for embeddings

 Embeds documents and stores them in ChromaDB

 Incremental updates – avoids re-embedding already processed rows

 Query ChromaDB collections with semantic similarity search


