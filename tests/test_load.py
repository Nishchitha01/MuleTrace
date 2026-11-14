import unittest
import pandas as pd
from src.load import load, time_split, STATIC_USD_RATES

class TestLoad(unittest.TestCase):
    def test_load_slice(self):
        df = load("data/HI-Small_Trans.csv", nrows=1000)
        self.assertEqual(len(df), 1000)
        
        # Check required columns
        expected_cols = [
            'ts', 'from_bank', 'src_acc', 'to_bank', 'dst_acc',
            'amt_recv', 'cur_recv', 'amount', 'cur_paid', 'fmt', 'label',
            'src', 'dst', 't', 'eid', 'amount_usd'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Check eid uniqueness and ordering
        self.assertTrue(df['t'].is_monotonic_increasing)
        self.assertEqual(list(df['eid']), list(range(1000)))
        
        # Check src and dst formatting
        self.assertTrue((df['src'] == df['from_bank'].astype(str) + '_' + df['src_acc'].astype(str)).all())
        self.assertTrue((df['dst'] == df['to_bank'].astype(str) + '_' + df['dst_acc'].astype(str)).all())
        
        # Check amount_usd calculation
        for idx, row in df.head(10).iterrows():
            expected_usd = row['amount'] * STATIC_USD_RATES[row['cur_paid']]
            self.assertAlmostEqual(row['amount_usd'], expected_usd, places=4)

    def test_time_split(self):
        df = load("data/HI-Small_Trans.csv", nrows=1000)
        train, val, test = time_split(df)
        self.assertEqual(len(train), 600)
        self.assertEqual(len(val), 200)
        self.assertEqual(len(test), 200)
        
        # Check temporal order
        self.assertLessEqual(train['t'].max(), val['t'].min())
        self.assertLessEqual(val['t'].max(), test['t'].min())

if __name__ == "__main__":
    unittest.main()
