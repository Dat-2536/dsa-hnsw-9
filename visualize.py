import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.spatial import distance
from hnsw import HNSWSearchSystem

# ==========================================
# 1. SETUP & MODERN STYLING
# ==========================================
# Color Palette (Flat UI)
COLORS = {
    'bg': '#f8f9fa',        # Light grey background
    'node_inactive': '#bdc3c7', # Grey
    'node_active': '#2980b9',   # Strong Blue
    'neighbor_good': '#27ae60', # Green
    'neighbor_bad': '#e74c3c',  # Red
    'query': '#e67e22',         # Orange
    'trail': '#3498db',         # Light Blue (History)
    'text': '#2c3e50',          # Dark Grey
    'true_best': '#f1c40f'      # Gold (True Nearest Neighbor)
}

print("Initializing HNSW System...")
# --- CHANGE DIMENSION HERE ---
dim = 2  # Works for 2, 3, 100, etc.
# -----------------------------

system = HNSWSearchSystem(space='l2', dim=dim)
system.build_hnsw_index(max_elements=1000, ef_construction=20, M=4)

np.random.seed(420)
num_elements = 400
data = np.random.rand(num_elements, dim).astype(np.float32)
ids = np.arange(num_elements)

system.add_items(data, ids)

# --- DYNAMIC QUERY POINT ---
query_point = np.full((dim,), 0.5, dtype=np.float32)

# --- FIND TRUE NEAREST NEIGHBOR (Brute Force on High-Dim Data) ---
print("Calculating Ground Truth...")
all_distances = [distance.euclidean(d, query_point) for d in data]
true_best_idx = np.argmin(all_distances)
true_best_id = ids[true_best_idx]
true_best_dist = all_distances[true_best_idx]
print(f"True Best Node: {true_best_id} at dist {true_best_dist:.4f}")

# ==========================================
# 2. PCA PROJECTION (The Fix for Visual Distortion)
# ==========================================
# If dim > 2, we project data to 2D using PCA so relative positions look better.
# If dim == 2, we just use the data as is.

# Combine data and query to find the best common plane
all_points = np.vstack([data, query_point])
data_mean = all_points.mean(axis=0)
centered_data = all_points - data_mean

if dim > 2:
    print(f"Projecting {dim}D data to 2D using PCA...")
    # SVD is a robust way to compute PCA without extra libraries like sklearn
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    # The first two rows of Vt are the principal components
    projection_matrix = Vt[:2, :].T
    projected_data = centered_data @ projection_matrix
else:
    projected_data = centered_data # In 2D, just center it

# Separate projected data back out
plot_data = projected_data[:-1] # All nodes
plot_query = projected_data[-1]  # The query point

# Helper to get PLOTTING coordinates (2D) from an ID
def get_plot_pos(id_val):
    return plot_data[int(id_val)]

# Helper to get LOGIC coordinates (N-Dim) for distance checks
def get_real_pos(id_val):
    return system.get_items([int(id_val)])[0]

# ==========================================
# 3. MAP NODES TO LAYERS
# ==========================================
max_graph_level = system.get_graph_max_level()
nodes_per_level = {i: [] for i in range(max_graph_level + 1)}
true_best_at_level = {i: False for i in range(max_graph_level + 1)}

for idx in ids:
    pos_2d = get_plot_pos(idx) # Store 2D pos for plotting
    lvl = system.get_element_max_level(int(idx))
    for l in range(lvl + 1):
        nodes_per_level[l].append(pos_2d)
        if int(idx) == int(true_best_id):
            true_best_at_level[l] = True

for l in nodes_per_level:
    nodes_per_level[l] = np.array(nodes_per_level[l])

# ==========================================
# 4. LOGIC RECORDING (High-Dimensional Physics)
# ==========================================
def record_search_events(system, query):
    events = [] 
    entry_point = system.get_entry_point()
    if entry_point is None: return []

    curr_node = entry_point
    curr_dist = distance.euclidean(get_real_pos(curr_node), query)
    
    for level in range(max_graph_level, -1, -1):
        events.append({"type": "LAYER_START", "level": level, "curr_node": curr_node, "dist": curr_dist})

        while True:
            try: neighbors = system.get_neighbors(curr_node, level)
            except: neighbors = []
            
            events.append({"type": "SCAN", "level": level, "curr_node": curr_node, "neighbors": neighbors, "dist": curr_dist})

            best_neighbor = curr_node
            best_neighbor_dist = curr_dist
            found_better = False

            for n_id in neighbors:
                # CRITICAL: Distance logic uses REAL HIGH-DIM positions
                d = distance.euclidean(get_real_pos(n_id), query)
                if d < best_neighbor_dist:
                    best_neighbor_dist = d
                    best_neighbor = n_id
                    found_better = True
            
            if found_better:
                events.append({
                    "type": "MOVE", "level": level, "curr_node": curr_node,
                    "from_node": curr_node, "to_node": best_neighbor, "dist": best_neighbor_dist
                })
                curr_node = best_neighbor
                curr_dist = best_neighbor_dist
            else:
                events.append({"type": "LOCAL_MIN", "level": level, "curr_node": curr_node, "dist": curr_dist})
                break 
        
        if level > 0:
            events.append({"type": "DROP_LAYER", "level": level, "next_level": level-1, "curr_node": curr_node, "dist": curr_dist})

    return events

raw_events = record_search_events(system, query_point)

# ==========================================
# 5. FRAME GENERATION (2D Projected View)
# ==========================================
animation_frames = []
history_trail = [] 

def interpolate(p1, p2, steps=10):
    return [p1 + (p2 - p1) * i / steps for i in range(steps + 1)]

for event in raw_events:
    # Use PLOT coordinates for visual history
    c_pos_plot = get_plot_pos(event['curr_node'])
    c_pos_real = get_real_pos(event['curr_node'])
    
    lvl = event['level']
    current_dist = event['dist']
    
    base_state = {
        'level': lvl, 'curr_pos': c_pos_plot, 'dist': current_dist,
        'neighbors_good': [], 'neighbors_bad': [],
        'trail': list(history_trail),
        'alpha': 1.0, 'status': 'normal'
    }

    if event['type'] == 'LAYER_START':
        for _ in range(10):
            frame = base_state.copy()
            frame.update({'title': f"Entering Layer {lvl}", 'status': 'entered'})
            animation_frames.append(frame)

    elif event['type'] == 'SCAN':
        n_ids = event['neighbors']
        good, bad = [], []
        for nid in n_ids:
            # Logic: Check Real Dist
            real_pos = get_real_pos(nid)
            # Visual: Append Plot Pos
            plot_pos = get_plot_pos(nid)
            
            if distance.euclidean(real_pos, query_point) < current_dist:
                good.append(plot_pos)
            else:
                bad.append(plot_pos)
        
        for _ in range(20):
            frame = base_state.copy()
            frame.update({
                'title': f"Scanning {len(n_ids)} Neighbors...", 
                'neighbors_good': good, 'neighbors_bad': bad,
                'status': 'scanning'
            })
            animation_frames.append(frame)

    elif event['type'] == 'MOVE':
        history_trail.append(c_pos_plot)
        start_plot = get_plot_pos(event['from_node'])
        end_plot = get_plot_pos(event['to_node'])
        
        path = interpolate(start_plot, end_plot, steps=20)
        for p in path:
            frame = base_state.copy()
            frame.update({
                'title': "Moving to Closer Neighbor", 
                'curr_pos': p, 
                'trail': list(history_trail),
                'status': 'moving'
            })
            animation_frames.append(frame)

    elif event['type'] == 'LOCAL_MIN':
        for _ in range(15):
            frame = base_state.copy()
            frame.update({'title': "Local Minimum Reached", 'status': 'stuck'})
            animation_frames.append(frame)

    elif event['type'] == 'DROP_LAYER':
        steps = 30
        for i in range(steps):
            frame = base_state.copy()
            frame.update({
                'title': f"Dropping Layer {lvl} -> {event['next_level']}",
                'status': 'fading',
                'next_level': event['next_level'],
                'fade_progress': i / steps
            })
            animation_frames.append(frame)

# ==========================================
# 6. PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(9, 9), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

def update(frame_idx):
    ax.clear()
    if not animation_frames: return

    state = animation_frames[frame_idx]
    
    # 1. Radar Circles (Only for 2D to avoid confusion)
    if dim == 2:
        for radius in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            circle = Circle((plot_query[0], plot_query[1]), radius, 
                            color=COLORS['query'], fill=False, alpha=0.1, linestyle='--')
            ax.add_patch(circle)

    # 2. Trail
    if state['trail']:
        trail_arr = np.array(state['trail'])
        ax.plot(trail_arr[:, 0], trail_arr[:, 1], color=COLORS['trail'], 
                linestyle=':', linewidth=2, alpha=0.5, label='History')
        ax.scatter(trail_arr[:, 0], trail_arr[:, 1], color=COLORS['trail'], s=30, alpha=0.4)

    # 3. Nodes (Fading or Normal)
    if state['status'] == 'fading':
        old_nodes = nodes_per_level[state['level']]
        if len(old_nodes):
            ax.scatter(old_nodes[:, 0], old_nodes[:, 1], c=COLORS['node_inactive'], s=60, 
                      alpha=1.0 - state['fade_progress'], edgecolors='white')
        new_nodes = nodes_per_level[state['next_level']]
        if len(new_nodes):
            ax.scatter(new_nodes[:, 0], new_nodes[:, 1], c=COLORS['node_inactive'], s=60, 
                      alpha=state['fade_progress'], edgecolors='white')
    else:
        nodes = nodes_per_level[state['level']]
        if len(nodes):
            ax.scatter(nodes[:, 0], nodes[:, 1], c=COLORS['node_inactive'], s=60, alpha=0.8, edgecolors='white')
        
        # True Best (Projected)
        if true_best_at_level[state['level']]:
            tb_plot = get_plot_pos(true_best_id)
            ax.scatter(tb_plot[0], tb_plot[1], c=COLORS['true_best'], marker='D', s=180, 
                      zorder=5, edgecolors='black', label='True Best')

    # 4. Neighbors
    if state['status'] == 'scanning':
        curr = state['curr_pos']
        if state['neighbors_bad']:
            bad = np.array(state['neighbors_bad'])
            ax.scatter(bad[:, 0], bad[:, 1], c=COLORS['neighbor_bad'], s=80, zorder=4, alpha=0.6)
            for n in bad:
                ax.plot([curr[0], n[0]], [curr[1], n[1]], color=COLORS['neighbor_bad'], linestyle=':', alpha=0.3)
        if state['neighbors_good']:
            good = np.array(state['neighbors_good'])
            ax.scatter(good[:, 0], good[:, 1], c=COLORS['neighbor_good'], s=120, zorder=5, edgecolors='white', linewidth=2)
            for n in good:
                ax.plot([curr[0], n[0]], [curr[1], n[1]], color=COLORS['neighbor_good'], linestyle='-', alpha=0.8, linewidth=2)

    # 5. Searcher & Query
    ax.scatter(state['curr_pos'][0], state['curr_pos'][1], c=COLORS['node_active'], s=250, zorder=6, edgecolors='white', linewidth=2)
    ax.scatter(plot_query[0], plot_query[1], c=COLORS['query'], marker='*', s=400, zorder=10, edgecolors='white', linewidth=1.5, label='Target')

    # Titles & Stats
    title_text = f"{state['title']}"
    if dim > 2:
        title_text += f" ({dim}D PCA -> 2D)"
        
    ax.set_title(title_text, fontsize=16, fontweight='bold', color=COLORS['text'])
    
    stats_text = (f"Current Dist: {state['dist']:.4f}\n"
                  f"True Best Dist: {true_best_dist:.4f}")
    
    ax.text(0.02, 0.92, stats_text, transform=ax.transAxes, 
            fontsize=12, color=COLORS['text'], 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', boxstyle='round,pad=0.5'))

    # Auto-scale axis to fit the projected data
    all_x = plot_data[:, 0]
    all_y = plot_data[:, 1]
    margin = 0.1
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=len(animation_frames), interval=40, repeat=False)
plt.show()