import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
collection = client['FaceRecProject']['PeopleMetadata']

# Xóa trường feature_vector trong TẤT CẢ các bản ghi
collection.update_many({}, {"$unset": {"feature_vector": ""}})

print("Đã xóa sạch dữ liệu vector cũ. Hãy chạy lại data_import.py!")
