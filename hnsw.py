import hnswlib
import numpy as np
import pickle

class HNSWSearchSystem:
    def __init__(self, space: str = 'l2', dim: int = 16):
        """
        Khởi tạo hệ thống tìm kiếm
        
        Args:
            space: Không gian khoảng cách ('l2', 'cosine', 'ip')
            dim: Số chiều của vector
        """
        self.space = space
        self.dim = dim
        self.index = hnswlib.Index(space, dim)
        self.is_built = False

    def build_hnsw_index(self, max_elements: int = 10000, ef_construction: int = 200, M: int = 128) -> None:
            """
            Xây dựng chỉ mục HNSW
            
            Args:
                data: Dữ liệu vector
                max_elements: Số lượng phần tử tối đa
                ef_construction: Tham số ef cho quá trình xây dựng 
                M: Tham số M (số lượng kết nối tối đa)
            """
            #xây dựng index
            self.max_elements = max_elements
            self.ef_construction = ef_construction
            self.M = M
            self.is_built = True
            self.index.init_index(max_elements, ef_construction, M)       

    def set_ef(self, val: int):
         # higher ef leads to better accuracy, but slower search
         self.ef_search = val
         self.index.set_ef(val)

    def set_num_threads(self, val: int):
         # Set number of threads used during batch search/construction
         # By default using all available cores
         self.num_threads = val
         self.index.set_num_threads(val)

    def add_items(self, items,  ids = None) -> None:
        """
        Thêm các phần tử vào self index
        """
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        self.index.add_items(items, ids)

    def get_items(self, ids):
         if isinstance(ids, int):
              ids = [ids]
         return self.index.get_items(ids)
    
    def get_dim(self):
         return self.dim
    
    def get_size(self):
         return self.index.get_current_count()
    
    def get_ids_list(self):
         return sorted(self.index.get_ids_list())
    
    def get_all_items(self):
         ids = self.get_ids_list()
         return self.get_items(ids)
    
    def set_items(self, items, ids):
         if isinstance(ids, int):
              ids = [ids]
         if (all(item in self.get_ids_list() for item in ids)): # for item in ids, if item also in self.ids.list then return true
              self.index.add_items(items, ids)
              return
         else:
              raise ValueError("ids is invalid")
         
    def delete_items(self,ids):
         if isinstance(ids, int):
              ids = [ids]
         for id in ids:
             self.index.mark_deleted(id)
     
    def clear(self) -> None:
          self.index = hnswlib.Index(self.space, self.dim)
          if hasattr(self, 'max_elements') and hasattr(self, 'ef_construction') and hasattr(self, 'M'):
               self.index.init_index(self.max_elements, self.ef_construction, self.M)
        
               # Thiết lập lại các tham số tìm kiếm nếu tồn tại
               if hasattr(self, 'ef_search'):
                    self.index.set_ef(self.ef_search)
               if hasattr(self, 'num_threads'):
                    self.index.set_num_threads(self.num_threads)
          else:
               # Nếu chưa từng được build, đánh dấu là chưa build
               self.is_built = False  
      
    def generate_data(self, num_elements:int) -> None:
         #Tạo ngẫu nhiên một só các vector để add vào đồ thị
         data = np.float32(np.random.random((num_elements, self.dim)))
         self.add_items(data)

    def knn_query(self, query: np.ndarray, k: int = 1) -> tuple:
        """
        Tìm kiếm K láng giềng gần nhất
        
        Args:
            query: Vector truy vấn (có thể là 1 vector hoặc nhiều vector)
            k: Số lượng kết quả trả về
        
        Returns:
            (labels, distances): Tuple chứa nhãn và khoảng cách của K kết quả gần nhất
        """
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        
        labels, distances = self.index.knn_query(query, k)
        return labels, distances
    
    def get_graph_max_level(self) -> int:
        """Trả về tầng cao nhất của toàn bộ đồ thị HNSW."""
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        return self.index.get_graph_max_level()

    def get_element_max_level(self, label: int) -> int:
        """Trả về tầng cao nhất mà một phần tử (theo label) tồn tại."""
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        return self.index.get_element_max_level(label)

    def get_neighbors(self, label: int, level: int) -> list:
        """
        Trả về danh sách các label hàng xóm của một phần tử tại một tầng (level) nhất định.
        """
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        return self.index.get_neighbors(label, level)

    def get_entry_point(self) -> int | None:
        """
        Trả về label của điểm vào (entry point) hiện tại của đồ thị.
        Trả về None nếu đồ thị rỗng.
        """
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        # Giả định hàm C++ binding có tên là get_entry_point_label()
        # và trả về None nếu không có entry point
        if hasattr(self.index, 'get_entry_point_label'):
             return self.index.get_entry_point_label()
        return None # Hoặc một giá trị mặc định nếu hàm chưa tồn tại
    
    
    def create_copy(self) -> 'HNSWSearchSystem':
        """
        Tạo bản Deep copy của hệ thống (dùng pickle round-trip)
        """
        # Tạo bản copy của index HNSW
        index_copy = pickle.loads(pickle.dumps(self.index))
        
        # Tạo hệ thống mới
        new_system = HNSWSearchSystem(self.space, self.dim)
        new_system.index = index_copy
        new_system.is_built = self.is_built
        new_system.max_elements = self.max_elements
        new_system.ef_construction = self.ef_construction
        new_system.M = self.M
        new_system.ef_search = self.ef_search
        
        return new_system