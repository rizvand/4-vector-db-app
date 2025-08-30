# Vector Database with Milvus and Custom Cosine Similarity

## How to Run

1. **Start Milvus with Docker:**
```bash
docker-compose up -d
```

2. **Create virtual environment and install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --only-binary=grpcio pymilvus
```

3. **Run the example:**
```bash
source venv/bin/activate
python milvus_db.py
```

## What it does

- Implements cosine similarity from scratch (no high-level libraries)
- Deploys Milvus vector database using Docker
- Compares manual calculation vs Milvus built-in similarity
- Shows both implementations produce identical results