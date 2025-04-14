🚀 Autonomous Campaign Builder - Embedding Pipeline
This project provides a pipeline to chunk, embed, and store tabular data from CSVs using either Google Gemini or OpenAI embeddings. It also stores and queries the embeddings using ChromaDB.Also it supports two major embedding providers – Google Gemini and OpenAI. We can switch between them by setting USE_OPENAI = True or False in the code.

📦 Features
 Load all CSVs from a folder structure (1 row = 1 chunk)

 Choose between OpenAI and Gemini for embeddings
 Model: models/embedding-001 (Gemini)
 Model: text-embedding-3-small(OpenAi)

 Embeds documents and stores them in ChromaDB

 Incremental updates – avoids re-embedding already processed rows

 Query ChromaDB collections with semantic similarity search


