# Vector Database with Milvus and Custom Cosine Similarity

## Setup

1. Start Milvus with Docker:
```bash
docker-compose up -d
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the example:
```bash
python milvus_db.py
```

## What it does

- Implements cosine similarity from scratch (no high-level libraries)
- Deploys Milvus vector database using Docker
- Compares manual calculation vs Milvus built-in similarity
- Shows both implementations work the same way