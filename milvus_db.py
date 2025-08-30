import math
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors from scratch"""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (magnitude_a * magnitude_b)


class MilvusVectorDB:
    def __init__(self, host='localhost', port='19530', collection_name='vectors'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.stored_vectors = []  # Keep copy for manual similarity calculation
        self.stored_labels = []
    
    def connect(self):
        """Connect to Milvus"""
        connections.connect("default", host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")
    
    def create_collection(self, dim=3):
        """Create collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=200)
        ]
        
        schema = CollectionSchema(fields, "Vector collection")
        
        if Collection.exists(self.collection_name):
            Collection(self.collection_name).drop()
        
        self.collection = Collection(self.collection_name, schema)
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 128}
        }
        self.collection.create_index("vector", index_params)
        print(f"Created collection: {self.collection_name}")
    
    def add_vectors(self, vectors, labels):
        """Add vectors to database"""
        self.stored_vectors.extend(vectors)
        self.stored_labels.extend(labels)
        
        data = [vectors, labels]
        self.collection.insert(data)
        self.collection.flush()
        print(f"Added {len(vectors)} vectors")
    
    def search_milvus(self, query_vector, top_k=5):
        """Search using Milvus built-in similarity"""
        self.collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            [query_vector], 
            "vector", 
            search_params, 
            limit=top_k, 
            output_fields=["label"]
        )
        
        return results[0]
    
    def search_manual(self, query_vector, top_k=5):
        """Search using our manual cosine similarity implementation"""
        results = []
        
        for i, vector in enumerate(self.stored_vectors):
            similarity = cosine_similarity(query_vector, vector)
            results.append({
                'label': self.stored_labels[i],
                'similarity': similarity,
                'vector': vector
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    # Initialize database
    db = MilvusVectorDB()
    db.connect()
    db.create_collection(dim=3)
    
    # Add example vectors
    vectors = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0]
    ]
    labels = ["doc1", "doc2", "doc3", "doc4"]
    
    db.add_vectors(vectors, labels)
    
    # Query vector
    query = [1.0, 2.0, 2.0]
    
    print(f"Query vector: {query}")
    print("\n--- Manual Cosine Similarity (from scratch) ---")
    manual_results = db.search_manual(query, top_k=3)
    for result in manual_results:
        print(f"{result['label']}: {result['similarity']:.3f}")
    
    print("\n--- Milvus Built-in Search ---")
    milvus_results = db.search_milvus(query, top_k=3)
    for result in milvus_results:
        print(f"{result.entity.get('label')}: {result.distance:.3f}")
    
    print("\n--- Manual Calculation Details ---")
    for i, vector in enumerate(vectors):
        similarity = cosine_similarity(query, vector)
        print(f"{labels[i]} {vector} -> similarity: {similarity:.3f}")