import unittest
import pandas as pd
from src.baseline import baseline, tune_baseline, evaluate_baseline

class TestBaseline(unittest.TestCase):
    def setUp(self):
        # Create a toy dataframe with known properties
        self.toy_df = pd.DataFrame({
            'ts': pd.to_datetime([
                '2022-09-01 01:00',
                '2022-09-01 02:00',
                '2022-09-01 03:00',
                '2022-09-01 04:00',
                '2022-09-02 01:00',
            ]),
            'src': ['ACC_A', 'ACC_A', 'ACC_B', 'ACC_C', 'ACC_A'],
            'amount_usd': [5000.0, 15000.0, 2000.0, 25000.0, 3000.0],
            'label': [0, 1, 0, 1, 0]
        })

    def test_amount_threshold(self):
        # Flag transactions with amount_usd > 10000
        flagged = baseline(self.toy_df, amt_thr=10000.0, daily_n=float('inf'), daily_sum=float('inf'))
        self.assertEqual(list(flagged), [False, True, False, True, False])

    def test_daily_n_threshold(self):
        # On 2022-09-01, ACC_A has 2 txs, ACC_B has 1, ACC_C has 1. On 09-02, ACC_A has 1.
        # daily_n > 1 flags ACC_A's two txs on 09-01
        flagged = baseline(self.toy_df, amt_thr=float('inf'), daily_n=1, daily_sum=float('inf'))
        self.assertEqual(list(flagged), [True, True, False, False, False])

    def test_daily_sum_threshold(self):
        # On 2022-09-01, ACC_A total is 20000. ACC_C is 25000.
        # daily_sum > 18000 flags ACC_A and ACC_C on 09-01
        flagged = baseline(self.toy_df, amt_thr=float('inf'), daily_n=float('inf'), daily_sum=18000.0)
        self.assertEqual(list(flagged), [True, True, False, True, False])

    def test_tune_baseline(self):
        best = tune_baseline(
            self.toy_df,
            amt_grid=[1000, 10000, 20000, float('inf')],
            n_grid=[1, 2, float('inf')],
            sum_grid=[1000, 10000, float('inf')]
        )
        self.assertIn('amt_thr', best)
        self.assertIn('daily_n', best)
        self.assertIn('daily_sum', best)
        self.assertGreater(best['f1'], 0.0)

    def test_evaluate_baseline(self):
        params = {'amt_thr': 10000.0, 'daily_n': float('inf'), 'daily_sum': float('inf')}
        metrics = evaluate_baseline(self.toy_df, params)
        # amt > 10000 flags indices 1 and 3, both have label==1 -> 100% precision & recall
        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(metrics['fp'], 0)
        self.assertEqual(metrics['fn'], 0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)

if __name__ == "__main__":
    unittest.main()
