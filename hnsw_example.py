from hnsw import HNSWSearchSystem
import numpy as np

system = HNSWSearchSystem(space='l2', dim = 4)
system.build_hnsw_index(max_elements=1000, ef_construction=200, M=16)

system.add_items([1,2,3,4])
system.add_items([2,3,5,3])
system.add_items([3,2,2,3])
system.add_items([4,4,4,4])
system.add_items([5,5,5,5])
system.add_items([[2,2,3,5], [4,2,5,6] , [10,4,2,1]])

print("size of the list: ", system.get_size())
print("ids list:", system.get_ids_list())
print("items in the list: \n", system.get_all_items())

query = [2,2,2,2]
k = 2
labels = system.knn_query(query, k)[0]
query_labels = labels[0]
print(f"{k} nearest neighbors of {query}: {system.get_items(query_labels)}")

queries = [[1,3,3,5], [3,4,2,1]]
labels = system.knn_query(queries, k)[0]
query1_labels = labels[0]
query2_labels = labels[1]
print(f"{k} nearest neighbors of {queries[0]}: ", system.get_items(query1_labels))
print(f"{k} nearest neighbors of {queries[1]}: ", system.get_items(query2_labels))