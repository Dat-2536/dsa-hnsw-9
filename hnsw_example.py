import unittest
import numpy as np
from hnsw import HNSWSearchSystem

class TestHNSWGraphStructure(unittest.TestCase):
    
    def setUp(self):
        """Run before every test: Setup a clean system"""
        self.dim = 16
        self.system = HNSWSearchSystem(space='l2', dim=self.dim)
        
    def test_01_empty_graph(self):
        """Test behavior when graph is initialized but empty"""
        print("\n--- Test 01: Empty Graph ---")
        self.system.build_hnsw_index(max_elements=100, M=16, ef_construction=100)
        
        ep = self.system.get_entry_point()
        print(f"Entry point for empty graph: {ep}")
        
        # Depending on bindings, this might be None or raise error
        # Just ensuring it doesn't crash the python script
        try:
            level = self.system.get_graph_max_level()
            print(f"Max level empty graph: {level}")
        except Exception as e:
            print(f"Caught expected exception for empty graph level: {e}")

    def test_02_single_node(self):
        """Test graph with exactly one node"""
        print("\n--- Test 02: Single Node ---")
        self.system.build_hnsw_index(max_elements=100, M=5, ef_construction=100)
        
        self.system.add_items(np.random.rand(1, self.dim), [0])
        
        ep = self.system.get_entry_point()
        self.assertEqual(ep, 0, "Entry point should be the only node 0")
        
        max_level = self.system.get_element_max_level(0)
        print(f"Node 0 is at max level: {max_level}")
        
        # Check neighbors at level 0 (should be empty as there is no one else)
        neighbors = self.system.get_neighbors(0, 0)
        self.assertEqual(len(neighbors), 0, "Single node should have 0 neighbors")

    def test_03_neighbor_constraints_M(self):
        """
        CRITICAL TEST: Verify M vs 2*M constraint.
        We set M=5. 
        Level > 0 should have <= 5 neighbors.
        Level 0 should have <= 10 neighbors.
        """
        print("\n--- Test 03: Neighbor Constraints (M=5) ---")
        M_val = 5
        self.system.build_hnsw_index(max_elements=1000, M=M_val, ef_construction=100)
        
        # Add enough data to force multiple layers
        num_elements = 200
        data = np.random.rand(num_elements, self.dim)
        ids = np.arange(num_elements)
        self.system.add_items(data, ids)
        
        print(f"Inserted {num_elements} elements.")
        
        violation_count = 0
        
        # Iterate over EVERY node in the graph
        for node_id in ids:
            # Get how high this node lives
            node_max_level = self.system.get_element_max_level(node_id)
            
            # Check neighbors at every level this node exists
            for level in range(node_max_level + 1):
                neighbors = self.system.get_neighbors(node_id, level)
                count = len(neighbors)
                
                if level == 0:
                    # Level 0 allows 2 * M
                    limit = 2 * M_val
                    if count > limit:
                        print(f"VIOLATION at Node {node_id} Level {level}: Has {count} neighbors (Limit {limit})")
                        violation_count += 1
                else:
                    # Upper levels allow M
                    limit = M_val
                    if count > limit:
                        print(f"VIOLATION at Node {node_id} Level {level}: Has {count} neighbors (Limit {limit})")
                        violation_count += 1
                        
        if violation_count == 0:
            print("SUCCESS: All nodes respect M and 2*M neighbor limits.")
        else:
            self.fail(f"Found {violation_count} constraint violations!")

    def test_04_connectivity(self):
        """Verify that neighbors point to valid existing nodes"""
        print("\n--- Test 04: Connectivity Check ---")
        self.system.build_hnsw_index(max_elements=100, M=16, ef_construction=100)
        
        ids = [10, 20, 30]
        self.system.add_items(np.random.rand(3, self.dim), ids)
        
        # Check node 10's neighbors
        neighbors = self.system.get_neighbors(10, 0)
        print(f"Node 10 neighbors: {neighbors}")
        
        # Ensure all neighbors are actually in our ID list
        for n in neighbors:
            self.assertIn(n, ids, f"Neighbor {n} is not a valid ID in the graph")

    def test_05_entry_point_reachability(self):
        """Verify we can reach the bottom layer from the entry point"""
        print("\n--- Test 05: Entry Point Reachability ---")
        self.system.build_hnsw_index(max_elements=500, M=16, ef_construction=100)
        self.system.add_items(np.random.rand(100, self.dim), np.arange(100))
        
        ep = self.system.get_entry_point()
        print(f"Entry Point: {ep}")
        
        self.assertIsNotNone(ep, "Entry point should not be None")
        
        # Validate entry point exists in max level
        graph_max = self.system.get_graph_max_level()
        ep_max = self.system.get_element_max_level(ep)
        
        print(f"Graph Max Level: {graph_max}")
        print(f"Entry Point Max Level: {ep_max}")
        
        # The entry point must exist at the top layer
        self.assertEqual(graph_max, ep_max, "Entry point must reside at the highest graph layer")

if __name__ == '__main__':
    unittest.main()