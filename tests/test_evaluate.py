import unittest
import pandas as pd
from src.evaluate import prf, account_level_prf, ring_recall, print_table

class TestEvaluate(unittest.TestCase):
    def setUp(self):
        """
        Create a tiny hand-made example that can be verified by hand:
        5 transactions total:
          eid 0: Legit (label 0), ACC_A -> ACC_B
          eid 1: Illicit (label 1), ACC_A -> ACC_C
          eid 2: Legit (label 0), ACC_B -> ACC_D
          eid 3: Illicit (label 1), ACC_C -> ACC_A
          eid 4: Legit (label 0), ACC_D -> ACC_B
          
        Total legit = 3 (eids 0, 2, 4)
        Total illicit = 2 (eids 1, 3)
        
        Accounts:
          Illicit accounts: {ACC_A, ACC_C} (participated in tx 1 or 3)
          Legit-only accounts: {ACC_B, ACC_D}
          Total accounts: 4
        """
        self.df = pd.DataFrame({
            'eid': [0, 1, 2, 3, 4],
            'src': ['ACC_A', 'ACC_A', 'ACC_B', 'ACC_C', 'ACC_D'],
            'dst': ['ACC_B', 'ACC_C', 'ACC_D', 'ACC_A', 'ACC_B'],
            'label': [0, 1, 0, 1, 0]
        })

    def test_transaction_prf_hand_made(self):
        # Flag transactions [1, 2]:
        # eid 1 is label 1 -> True Positive (TP = 1)
        # eid 2 is label 0 -> False Positive (FP = 1)
        # eid 3 is label 1 and unflagged -> False Negative (FN = 1)
        # eids 0, 4 are label 0 and unflagged -> True Negative (TN = 2)
        flagged = [1, 2]
        res = prf(flagged, self.df)
        
        self.assertEqual(res['tp'], 1)
        self.assertEqual(res['fp'], 1)
        self.assertEqual(res['fn'], 1)
        self.assertEqual(res['tn'], 2)
        
        # Precision = 1 / (1 + 1) = 0.50
        # Recall = 1 / (1 + 1) = 0.50
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.50
        self.assertAlmostEqual(res['precision'], 0.5)
        self.assertAlmostEqual(res['recall'], 0.5)
        self.assertAlmostEqual(res['f1'], 0.5)

    def test_account_level_prf_hand_made(self):
        # Ground truth illicit accounts are {ACC_A, ACC_C}.
        # Flagged accounts: {ACC_A, ACC_B}
        # ACC_A is illicit and flagged -> TP = 1
        # ACC_B is legit and flagged -> FP = 1
        # ACC_C is illicit and unflagged -> FN = 1
        # ACC_D is legit and unflagged -> TN = 1
        flagged_accs = ['ACC_A', 'ACC_B']
        res = account_level_prf(flagged_accs, self.df)
        
        self.assertEqual(res['tp'], 1)
        self.assertEqual(res['fp'], 1)
        self.assertEqual(res['fn'], 1)
        self.assertEqual(res['tn'], 1)
        
        # Precision = 1 / (1 + 1) = 0.50
        # Recall = 1 / (1 + 1) = 0.50
        # F1 = 0.50
        self.assertAlmostEqual(res['precision'], 0.5)
        self.assertAlmostEqual(res['recall'], 0.5)
        self.assertAlmostEqual(res['f1'], 0.5)

    def test_ring_recall_hand_made(self):
        # Hand-made rings:
        # Ring 1: CYCLE with 3 accounts {ACC_A, ACC_B, ACC_C}
        # Ring 2: FAN-OUT with 4 accounts {ACC_D, ACC_E, ACC_F, ACC_G}
        rings = [
            {'type': 'CYCLE', 'accounts': {'ACC_A', 'ACC_B', 'ACC_C'}},
            {'type': 'FAN-OUT', 'accounts': {'ACC_D', 'ACC_E', 'ACC_F', 'ACC_G'}}
        ]
        
        # Hand-made alerts:
        # Alert 1: contains {ACC_A, ACC_B, ACC_C} -> 3/3 = 100% (>= 70%) of Ring 1 -> FOUND
        # Alert 2: contains {ACC_D, ACC_E} -> 2/4 = 50% (< 70%) of Ring 2 -> NOT FOUND
        alerts = [
            {'alert_id': 'A-01', 'accounts': ['ACC_A', 'ACC_B', 'ACC_C']},
            {'alert_id': 'A-02', 'accounts': ['ACC_D', 'ACC_E']}
        ]
        
        res = ring_recall(alerts, rings)
        self.assertEqual(res['found'], 1)
        self.assertEqual(res['total'], 2)
        self.assertAlmostEqual(res['overall_recall'], 0.50)
        
        self.assertEqual(res['per_type']['CYCLE']['found'], 1)
        self.assertEqual(res['per_type']['CYCLE']['total'], 1)
        self.assertAlmostEqual(res['per_type']['CYCLE']['recall'], 1.0)
        
        self.assertEqual(res['per_type']['FAN-OUT']['found'], 0)
        self.assertEqual(res['per_type']['FAN-OUT']['total'], 1)
        self.assertAlmostEqual(res['per_type']['FAN-OUT']['recall'], 0.0)

if __name__ == "__main__":
    unittest.main()
