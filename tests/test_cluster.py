import unittest
import pandas as pd
from src.cluster import merge_hits

class TestCluster(unittest.TestCase):
    def setUp(self):
        # Sample dataframe with 4 edges
        # Edge 0: A -> B, t=1.0, amt=100
        # Edge 1: B -> C, t=2.0, amt=90
        # Edge 2: C -> D, t=3.0, amt=80
        # Edge 3: D -> E, t=4.0, amt=70
        data = [
            {"eid": 0, "src": "1_A", "dst": "1_B", "t": 1.0, "amount_usd": 100.0},
            {"eid": 1, "src": "1_B", "dst": "1_C", "t": 2.0, "amount_usd": 90.0},
            {"eid": 2, "src": "1_C", "dst": "1_D", "t": 3.0, "amount_usd": 80.0},
            {"eid": 3, "src": "1_D", "dst": "1_E", "t": 4.0, "amount_usd": 70.0},
        ]
        self.df = pd.DataFrame(data)

    def test_no_merge_on_account_only(self):
        # Hit 1 has edge 0 (accounts A, B)
        # Hit 2 has edge 1 (accounts B, C)
        # They share account B, but NO shared eid!
        hits = [
            {"pattern": "fan_out", "eids": {0}, "hub": "1_A"},
            {"pattern": "fan_in", "eids": {1}, "hub": "1_C"},
        ]
        clusters = merge_hits(hits, self.df)
        # Must produce 2 separate clusters!
        self.assertEqual(len(clusters), 2)
        c0_eids = clusters[0]["eids"]
        c1_eids = clusters[1]["eids"]
        self.assertTrue((c0_eids == {0} and c1_eids == {1}) or (c0_eids == {1} and c1_eids == {0}))

    def test_merge_on_shared_edge(self):
        # Hit 1 has edges {0, 1}
        # Hit 2 has edges {1, 2}
        # They share edge 1!
        hits = [
            {"pattern": "fan_out", "eids": {0, 1}, "hub": "1_A"},
            {"pattern": "chain", "eids": {1, 2}, "hub": "1_B"},
        ]
        clusters = merge_hits(hits, self.df)
        # Must merge into 1 cluster containing {0, 1, 2}
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["eids"], {0, 1, 2})
        self.assertEqual(clusters[0]["patterns"], {"fan_out", "chain"})

    def test_split_if_over_max_accounts(self):
        # Hit 1 has edges {0, 1}
        # Hit 2 has edges {1, 2}
        # Merged would have 4 accounts (A, B, C, D).
        # With max_accounts=3, it should split back into 2 original hits!
        hits = [
            {"pattern": "fan_out", "eids": {0, 1}, "hub": "1_A"},
            {"pattern": "chain", "eids": {1, 2}, "hub": "1_B"},
        ]
        clusters = merge_hits(hits, self.df, max_accounts=3)
        self.assertEqual(len(clusters), 2)

if __name__ == "__main__":
    unittest.main()
