import numpy as np
from pymongo import MongoClient
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


try:
    from hnsw import HNSWSearchSystem
except ImportError:
        print("[WARN] Không tìm thấy module 'hnsw'")

from dotenv import load_dotenv
load_dotenv()

class FaceSearchEngine:
    def __init__(self):
        # Configure MongoDB connection
        self.uri = os.getenv("MONGO_URI")
        if not self.uri:
             raise ValueError("Chưa cấu hình MONGO_URI trong file .env")
        self.client = MongoClient(self.uri)
        self.collection = self.client['FaceRecProject']['PeopleMetadata']
        
        # Initialize the search system
        self.dim = 128
        self.search_system = None
        
        # Prefer the wrapper class if available; otherwise fall back to the core library
        if HNSWSearchSystem:
            self.search_system = HNSWSearchSystem(space='l2', dim=self.dim)
        else:
            # Fallback to the original hnswlib when the wrapper is missing
            import hnswlib
            self.search_system = hnswlib.Index(space='l2', dim=self.dim)

        self.metadata_mapping = {}
    
    def load_data_and_build_index(self):
        print("Đang tải dữ liệu vector từ MongoDB...")
        cursor = self.collection.find({"feature_vector": {"$exists": True}})
        
        vectors = []
        ids = []
        self.metadata_mapping = {}
        current_id = 0
        
        for doc in cursor:
            vec = doc['feature_vector']
            
            if isinstance(vec, list) and len(vec) == 128:
                vectors.append(vec)
                ids.append(current_id)
                
                self.metadata_mapping[current_id] = {
                    "MSSV": doc.get("MSSV", "Unknown"),
                    "Ten": doc.get("Ten", "Unknown"),
                    "MongoID": str(doc["_id"])
                }
                current_id += 1
            
        if len(vectors) == 0:
            print("Database rỗng hoặc chưa chạy data_import.py!")
            return

        print(f"Đã tải {len(vectors)} vector. Đang xây dựng HNSW Index...")

        # Build the index using methods on HNSWSearchSystem
        # Note: the wrapper exposes build_hnsw_index and add_items
        if hasattr(self.search_system, 'build_hnsw_index'):
             self.search_system.build_hnsw_index(
                max_elements=len(vectors) + 1000, 
                ef_construction=200, 
                M=16
            )
             self.search_system.add_items(np.array(vectors), np.array(ids))
             self.search_system.set_ef(50)
        else:
            # Fallback for the base hnswlib library when the wrapper is not used
            self.search_system.init_index(max_elements=len(vectors) + 1000, ef_construction=200, M=16)
            self.search_system.add_items(np.array(vectors), np.array(ids))
            self.search_system.set_ef(50)
        
        # Derive index size (wrapper has get_size, base library uses get_current_count)
        size = self.search_system.get_size() if hasattr(self.search_system, 'get_size') else self.search_system.get_current_count()
        print(f"Xây dựng xong Index với {size} phần tử!")
    
    def search_face(self, query_vector):
        """
        Input: query_vector (list or numpy array, length 128)
        """
        # Verify the index is built
        is_built = False
        if hasattr(self.search_system, 'is_built'):
            is_built = self.search_system.is_built
        else:
            # The base library treats init as built
            is_built = True 

        if self.search_system is None or not is_built:
            print("Lỗi: Chưa build index.")
            return None

        # Query
        query_np = np.array([query_vector])
        
        try:
            # Use knn_query from the wrapper or the base library
            labels, distances = self.search_system.knn_query(query_np, k=1)
            
            found_id = labels[0][0]
            distance = distances[0][0]
            
            # Threshold: 0.5 - 0.6 is a typical range for Euclidean distance (l2)
            if distance > 0.5: 
                return {"status": "unknown", "distance": float(distance)}
                
            info = self.metadata_mapping.get(found_id)
            return {
                "status": "found",
                "info": info,
                "distance": float(distance)
            }
        except Exception as e:
            print(f"Lỗi khi search vector: {e}")
            return None

# --- Test harness ---
if __name__ == "__main__":
    engine = FaceSearchEngine()
    engine.load_data_and_build_index()
    
    # Quick sanity check with a random vector
    dummy_vec = np.random.rand(128)
    print("Test Search Result:", engine.search_face(dummy_vec))