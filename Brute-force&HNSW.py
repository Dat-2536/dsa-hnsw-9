import numpy as np
import time
import hnswlib
import matplotlib.pyplot as plt

# (1) LỚP BRUTE-FORCE KNN
class BruteForceKNN:
    def __init__(self):
        self.data = None
        self.dim = 0

    def fit(self, data: np.ndarray):
        self.data = data
        self.dim = data.shape[1]

    def search(self, query_vector: np.ndarray, k: int = 5):
        distances = np.linalg.norm(self.data - query_vector, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        return top_k_indices, distances[top_k_indices]


# (2) LỚP HNSW SEARCH SYSTEM
class HNSWSearchSystem:
    def __init__(self, space: str = 'l2', dim: int = 16):
        self.space = space
        self.dim = dim
        self.index = hnswlib.Index(space, dim)
        self.is_built = False

    def build_hnsw_index(self, max_elements: int, ef_construction: int = 200, M: int = 16):
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.is_built = True

    def set_ef(self, val: int):
        self.index.set_ef(val)

    def add_items(self, items,  ids = None) -> None:
        """
        Thêm các phần tử vào self index
        """
        if not self.is_built:
            raise ValueError("Chưa build index! Gọi build_hnsw_index() trước.")
        self.index.add_items(items, ids)

    def knn_query(self, query: np.ndarray, k: int = 5):
        labels, distances = self.index.knn_query(query, k)
        return labels, distances


# (3) HÀM SO SÁNH 1 LẦN
def main():
    N_SAMPLES = 1000
    DIMENSION = 128
    K_NEIGHBORS = 10

    print(f"Tạo {N_SAMPLES} vector {DIMENSION} chiều...")
    data = np.random.rand(N_SAMPLES, DIMENSION).astype(np.float32)
    query_idx = np.random.randint(0, N_SAMPLES)
    query_vec = data[query_idx]
    print(f"Chọn vector truy vấn: chỉ số {query_idx}")

    print("\n Đang chạy BruteForceKNN...")
    bf = BruteForceKNN()
    bf.fit(data)
    start = time.time()
    bf_indices, bf_distances = bf.search(query_vec, k=K_NEIGHBORS)
    end = time.time()
    bf_time = (end - start) * 1000
    print(f" Thời gian (Brute-force): {bf_time:.4f} ms")

    print("\n Đang chạy HNSW Search...")
    hnsw = HNSWSearchSystem(space='l2', dim=DIMENSION)
    hnsw.build_hnsw_index(max_elements=N_SAMPLES)
    hnsw.set_ef(100)
    hnsw.add_items(data)
    start = time.time()
    labels, distances = hnsw.knn_query(query_vec, k=K_NEIGHBORS)
    end = time.time()
    hnsw_time = (end - start) * 1000
    print(f" Thời gian (HNSW): {hnsw_time:.4f} ms")

    print("\n===  So sánh kết quả ===")
    print(f"Top 10 (Brute-force): {bf_indices}")
    print(f"Top 10 (HNSW):       {labels[0]}")
    overlap = len(set(bf_indices).intersection(set(labels[0])))
    print(f"\n Số lượng phần tử trùng nhau trong Top-K: {overlap}/{K_NEIGHBORS}")
    print(f" HNSW nhanh hơn khoảng: {bf_time / hnsw_time:.2f} lần (ước tính)")


# (4) HÀM ĐO HIỆU NĂNG & VẼ ĐỒ THỊ THỜI GIAN
def benchmark_performance():
    DIM = 128
    K = 10
    sizes = [500, 1000, 2000, 5000, 10000]
    num_queries = 10

    bf_times, hnsw_times = [], []

    for n in sizes:
        print(f"\n Đang kiểm tra với {n} vector...")
        data = np.random.rand(n, DIM).astype(np.float32)
        queries = data[np.random.choice(n, num_queries, replace=False)]

        # Brute-force
        bf = BruteForceKNN()
        bf.fit(data)
        t0 = time.perf_counter()
        for q in queries:
            bf.search(q, K)
        t1 = time.perf_counter()
        bf_time = (t1 - t0) / num_queries * 1000
        bf_times.append(bf_time)

        # HNSW
        hnsw = HNSWSearchSystem(space='l2', dim=DIM)
        hnsw.build_hnsw_index(max_elements=n)
        hnsw.set_ef(100)
        hnsw.add_items(data)
        t0 = time.perf_counter()
        hnsw.knn_query(queries, K)
        t1 = time.perf_counter()
        hnsw_time = (t1 - t0) / num_queries * 1000
        hnsw_times.append(hnsw_time)

        print(f" Brute-force: {bf_time:.4f} ms | HNSW: {hnsw_time:.4f} ms")

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, bf_times, 'o-', label='Brute-force (chính xác)')
    plt.plot(sizes, hnsw_times, 's-', label='HNSW (xấp xỉ)')
    plt.xlabel('Số lượng vector dữ liệu (N)')
    plt.ylabel('Thời gian trung bình (ms/truy vấn)')
    plt.title('So sánh tốc độ: Brute-force vs HNSW')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# (5) BIỂU ĐỒ RECALL@K vs THỜI GIAN
def recall_vs_time():
    N = 10000
    DIM = 128
    K = 10
    num_queries = 50
    ef_values = [10, 30, 50, 100, 200, 400, 800]

    data = np.random.rand(N, DIM).astype(np.float32)
    queries = data[np.random.choice(N, num_queries, replace=False)]

    bf = BruteForceKNN()
    bf.fit(data)
    bf_results = [bf.search(q, K)[0] for q in queries]

    recalls, times = [], []

    for ef in ef_values:
        hnsw = HNSWSearchSystem(space='l2', dim=DIM)
        hnsw.build_hnsw_index(max_elements=N)
        hnsw.set_ef(ef)
        hnsw.add_items(data)

        start = time.time()
        hnsw_results, _ = hnsw.knn_query(queries, k=K)
        end = time.time()
        avg_time = (end - start) / num_queries * 1000
        times.append(avg_time)

        overlap_total = sum(len(set(bf_results[i]).intersection(set(hnsw_results[i]))) for i in range(num_queries))
        recall = overlap_total / (num_queries * K)
        recalls.append(recall)
        print(f"ef={ef:<4} | Recall@{K}: {recall:.3f} | Thời gian: {avg_time:.3f} ms/truy vấn")

    plt.figure(figsize=(8, 5))
    plt.plot(times, recalls, 'o-r', linewidth=2)
    plt.xlabel('Thời gian trung bình (ms/truy vấn)')
    plt.ylabel(f'Recall@{K}')
    plt.title(f'Mối quan hệ giữa Recall@{K} và Thời gian (HNSW, N={N})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# (6) CHẠY TOÀN BỘ CHƯƠNG TRÌNH
if __name__ == "__main__":
    main()
    benchmark_performance()
    recall_vs_time()
