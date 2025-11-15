# visualize_final.py (Sửa lỗi màu sắc và vị trí tiêu đề)

import networkx as nx
import plotly.graph_objects as go
import math

# Import lớp HNSWSearchSystem từ file hnsw.py của bạn
from hnsw import HNSWSearchSystem

# ==============================================================================
# 1. TẠO DỮ LIỆU HNSW
# ==============================================================================
print("Đang khởi tạo và xây dựng HNSW index...")
system = HNSWSearchSystem(space='l2', dim=4)
M = 2
EF_CONSTRUCTION = 4
NUM_ELEMENTS = 200

system.build_hnsw_index(max_elements=NUM_ELEMENTS + 10, ef_construction=EF_CONSTRUCTION, M=M)
system.generate_data(NUM_ELEMENTS)
print(f"Index được tạo với {system.get_size()} phần tử.")

# ==============================================================================
# 2. TRÍCH XUẤT DỮ LIỆU VÀ XÂY DỰNG ĐỒ THỊ NETWORKX
# (Phần này giữ nguyên, logic đã chính xác)
# ==============================================================================
print("Đang trích xuất dữ liệu và xây dựng đồ thị NetworkX...")
G = nx.Graph()
current_labels = system.get_ids_list()
max_level = system.get_graph_max_level()
LEVEL_SPACING = 30

# --- Xác định tầng cao nhất của mỗi node ---
node_info_map = {}
for label in current_labels:
    node_max_level = -1
    for l in range(max_level, -1, -1):
        if system.get_neighbors(label, l) or (label == system.get_entry_point() and l == max_level):
            node_max_level = l
            break
    node_info_map[label] = {"id": int(label), "level": node_max_level}

# --- Tính toán vị trí (x, z) chỉ dựa trên Layer 0 ---
label_to_coords = {}
all_nodes_info = list(node_info_map.values())
count = len(all_nodes_info)
radius = 6 + count * 0.4
angle_step = 2 * math.pi / count if count > 0 else 0
for i, info in enumerate(all_nodes_info):
    angle = i * angle_step
    x, z = radius * math.cos(angle), radius * math.sin(angle)
    label_to_coords[info['id']] = {'x': x, 'z': z}

# --- Thêm node vào đồ thị NetworkX ---
total_height = max_level * LEVEL_SPACING
for label, info in node_info_map.items():
    coords = label_to_coords.get(label)
    pos_y = (info['level'] * LEVEL_SPACING) - (total_height / 2)
    G.add_node(label, level=info['level'], pos_x=coords['x'], pos_y=pos_y, pos_z=coords['z'], is_entry_point=(label == system.get_entry_point()))

# --- Thêm cạnh vào đồ thị NetworkX ---
for level in range(max_level + 1):
    for label in current_labels:
        if G.nodes[label]['level'] >= level:
            neighbors = system.get_neighbors(label, level)
            for neighbor_label in neighbors:
                if int(label) < int(neighbor_label):
                    G.add_edge(label, neighbor_label, level=level, type='in-layer')
for label, info in node_info_map.items():
    if info['level'] > 0:
        query_vector = system.get_items(label)
        neighbor_labels, _ = system.knn_query(query_vector, k=1)
        target_label = neighbor_labels[0][0]
        if label != target_label:
            G.add_edge(label, target_label, level=-1, type='inter-layer')

# ==============================================================================
# 3. TRỰC QUAN HÓA BẰNG PLOTLY (CẬP NHẬT STYLE)
# ==============================================================================
print("Đang tạo hình ảnh trực quan 3D tương tác bằng Plotly...")
all_traces = []

# --- Tạo các "dấu vết" (traces) cho chế độ xem "All Layers" ---
edge_x, edge_y, edge_z = [], [], []
for edge in G.edges(data=True):
    if edge[2]['type'] == 'in-layer':
        x0, y0, z0, x1, y1, z1 = G.nodes[edge[0]]['pos_x'], G.nodes[edge[0]]['pos_y'], G.nodes[edge[0]]['pos_z'], G.nodes[edge[1]]['pos_x'], G.nodes[edge[1]]['pos_y'], G.nodes[edge[1]]['pos_z']
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None]); edge_z.extend([z0, z1, None])
all_traces.append(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=0.8, color='#aaa'), hoverinfo='none'))

inter_edge_x, inter_edge_y, inter_edge_z = [], [], []
for edge in G.edges(data=True):
    if edge[2]['type'] == 'inter-layer':
        x0, y0, z0, x1, y1, z1 = G.nodes[edge[0]]['pos_x'], G.nodes[edge[0]]['pos_y'], G.nodes[edge[0]]['pos_z'], G.nodes[edge[1]]['pos_x'], G.nodes[edge[1]]['pos_y'], G.nodes[edge[1]]['pos_z']
        inter_edge_x.extend([x0, x1, None]); inter_edge_y.extend([y0, y1, None]); inter_edge_z.extend([z0, z1, None])
all_traces.append(go.Scatter3d(x=inter_edge_x, y=inter_edge_y, z=inter_edge_z, mode='lines', line=dict(width=2, color='red'), hoverinfo='none'))

node_x, node_y, node_z, node_text, node_colors, node_sizes = [], [], [], [], [], []
for node, attr in G.nodes(data=True):
    node_x.append(attr['pos_x']); node_y.append(attr['pos_y']); node_z.append(attr['pos_z'])
    is_entry = attr['is_entry_point']
    entry_txt = "<b>(ENTRY POINT)</b>" if is_entry else ""
    node_text.append(f"ID: {node}<br>Max Level: {attr['level']}<br>{entry_txt}")
    node_colors.append(attr['level']); node_sizes.append(15 if is_entry else 8)
all_traces.append(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', text=node_text, hoverinfo='text', 
    marker=dict(
        size=node_sizes, 
        color=node_colors, 
        colorscale='Viridis',
        # **SỬA LỖI 1**: Khóa thang màu
        cmin=0,
        cmax=max_level,
        showscale=True, 
        colorbar=dict(
            thickness=15, 
            # **SỬA LỖI 2**: Di chuyển tiêu đề sang phải
            title=dict(text='Node Level', side='right')
        ),
        line=dict(width=2, color='DarkSlateGrey')
    )
))

# --- Tạo các "dấu vết" ẩn cho từng tầng cụ thể ---
for level in range(max_level + 1):
    edge_x_l, edge_y_l, edge_z_l = [], [], []
    node_x_l, node_y_l, node_z_l, node_text_l, node_sizes_l, node_colors_l = [], [], [], [], [], []
    for edge in G.edges(data=True):
        if edge[2]['level'] == level:
            x0, y0, z0, x1, y1, z1 = G.nodes[edge[0]]['pos_x'], G.nodes[edge[0]]['pos_y'], G.nodes[edge[0]]['pos_z'], G.nodes[edge[1]]['pos_x'], G.nodes[edge[1]]['pos_y'], G.nodes[edge[1]]['pos_z']
            edge_x_l.extend([x0, x1, None]); edge_y_l.extend([y0, y1, None]); edge_z_l.extend([z0, z1, None])
    for node, attr in G.nodes(data=True):
        if attr['level'] >= level:
            node_x_l.append(attr['pos_x']); node_y_l.append(attr['pos_y']); node_z_l.append(attr['pos_z'])
            is_entry = attr['is_entry_point']
            entry_txt = "<b>(ENTRY POINT)</b>" if is_entry else ""
            node_text_l.append(f"ID: {node}<br>Max Level: {attr['level']}<br>{entry_txt}")
            node_colors_l.append(attr['level']); node_sizes_l.append(15 if is_entry else 8)
    all_traces.append(go.Scatter3d(x=edge_x_l, y=edge_y_l, z=edge_z_l, mode='lines', line=dict(width=0.8, color='#aaa'), hoverinfo='none', visible=False))
    all_traces.append(go.Scatter3d(x=node_x_l, y=node_y_l, z=node_z_l, mode='markers', text=node_text_l, hoverinfo='text', 
        marker=dict(
            size=node_sizes_l, 
            color=node_colors_l, 
            colorscale='Viridis',
            # **SỬA LỖI 1**: Khóa thang màu (cho cả các view layer)
            cmin=0,
            cmax=max_level,
            showscale=False,
            line=dict(width=2, color='DarkSlateGrey')
        ), 
        visible=False
    ))

# --- Tạo Figure và Dropdown Menu ---
fig = go.Figure(data=all_traces)
buttons = []
visibility_all = [True, True, True] + [False] * (len(all_traces) - 3)
buttons.append(dict(label="All Layers", method="restyle", args=[{"visible": visibility_all}]))
for level in range(max_level + 1):
    visibility_level = [False] * len(all_traces)
    visibility_level[3 + level * 2] = True
    visibility_level[3 + level * 2 + 1] = True
    buttons.append(dict(label=f"Layer {level}", method="restyle", args=[{"visible": visibility_level}]))

fig.update_layout(
    updatemenus=[dict(
        active=0, buttons=buttons, direction="down",
        pad={"r": 10, "t": 10}, showactive=True,
        x=0.05, xanchor="left", y=1.1, yanchor="top"
    )],
    title='Trực quan hóa HNSW 3D',
    scene=dict(
        xaxis=dict(showbackground=False, showticklabels=False, title=''),
        yaxis=dict(showbackground=False, showticklabels=False, title='Layer Height'),
        zaxis=dict(showbackground=False, showticklabels=False, title=''),
        camera=dict(eye=dict(x=1.2, y=2.0, z=1.2))
    ),
    margin=dict(t=50, b=0, l=0, r=0)
)

print("Đang mở đồ thị trong trình duyệt...")
fig.show()
print("Hoàn tất!")