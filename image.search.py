from hnsw import HNSWSearchSystem
import numpy as np
from datasets import load_dataset
from base64 import b64decode
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests

data = load_dataset('food101', split='train', streaming=True).take(1000)
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_vector(image, bins=32):
    red = cv2.calcHist(
        [image], [2], None, [bins], [0, 256]
    )
    green = cv2.calcHist(
        [image], [1], None, [bins], [0, 256]
    )
    blue = cv2.calcHist(
        [image], [0], None, [bins], [0, 256]
    )
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    return vector

def imread_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image_array = np.frombuffer(response.content, np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        return image

    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None
    
images = [pil_to_cv2(sample['image']) for sample in data]

bins = 32
imageDataBase = HNSWSearchSystem(space='l2', dim = bins*3)
# Build index với các tham số phù hợp cho ảnh
imageDataBase.build_hnsw_index(
    max_elements=100000,  # Số lượng ảnh tối đa
    ef_construction=20,  # Trade-off build time vs quality
    M=16              # Trade-off memory vs accuracy
)
# Thiết lập ef cho search
imageDataBase.set_ef(10)  # Tăng ef để có độ chính xác cao hơn

# thêm vector đặc trưng ảnh vào đồ thị
for image in images:
    imageDataBase.add_items(get_vector(image,bins))

query_image1 = images[28]
query_image2 = imread_from_url("https://cdn.pixabay.com/photo/2020/10/17/11/06/pizza-5661748_1280.jpg")
if (query_image2 is None):
    raise ValueError("Không tìm thấy file trong đường dẫn")
query = [get_vector(query_image1, bins), get_vector(query_image2, bins)]
k = 4
labels = imageDataBase.knn_query(query, k)[0]
result_labels1 = labels[0]
result_labels2 = labels[1]

print("size:",imageDataBase.get_size())
# === FIGURE 1: CHO TRUY VẤN THỨ NHẤT ===
fig1, axes1 = plt.subplots(1, k + 1, figsize=(20, 4))
fig1.suptitle('Kết quả cho Ảnh truy vấn 1 ', fontsize=16)

# Hiển thị ảnh truy vấn 1
axes1[0].imshow(cv2.cvtColor(query_image1, cv2.COLOR_BGR2RGB))
axes1[0].set_title("Ảnh truy vấn 1")
axes1[0].axis('off')

# Hiển thị các kết quả cho truy vấn 1
for i in range(k):
    result_idx = result_labels1[i]
    axes1[i+1].imshow(cv2.cvtColor(images[result_idx], cv2.COLOR_BGR2RGB))
    axes1[i+1].axis('off')

# === FIGURE 2: CHO TRUY VẤN THỨ HAI ===
# Gọi plt.subplots() lần nữa sẽ tạo một figure thứ hai hoàn toàn mới
fig2, axes2 = plt.subplots(1, k + 1, figsize=(20, 4))
fig2.suptitle('Kết quả cho Ảnh truy vấn 2', fontsize=16)

# Hiển thị ảnh truy vấn 2
axes2[0].imshow(cv2.cvtColor(query_image2, cv2.COLOR_BGR2RGB))
axes2[0].set_title("Ảnh truy vấn 2")
axes2[0].axis('off')

# Hiển thị các kết quả cho truy vấn 2
for i in range(k):
    result_idx = result_labels2[i]
    axes2[i+1].imshow(cv2.cvtColor(images[result_idx], cv2.COLOR_BGR2RGB))
    axes2[i+1].axis('off')

# Lệnh này sẽ hiển thị TẤT CẢ các figure đã được tạo (fig1 và fig2)
plt.show()