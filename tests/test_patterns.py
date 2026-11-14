import unittest
import os
import tempfile
import pandas as pd
from src.patterns import parse_patterns, count_rings_by_type

class TestPatterns(unittest.TestCase):
    def test_parse_sample(self):
        sample_text = """
BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 2
2022/09/01 01:00,001,ACC1,002,ACC2,100.0,US Dollar,100.0,US Dollar,ACH,1
2022/09/01 02:00,001,ACC1,003,ACC3,150.0,US Dollar,150.0,US Dollar,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE: 3 hops
2022/09/01 03:00,001,ACC1,002,ACC2,200.0,US Dollar,200.0,US Dollar,ACH,1
2022/09/01 04:00,002,ACC2,003,ACC3,200.0,US Dollar,200.0,US Dollar,ACH,1
2022/09/01 05:00,003,ACC3,001,ACC1,200.0,US Dollar,200.0,US Dollar,ACH,1
END LAUNDERING ATTEMPT - CYCLE
"""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write(sample_text)
            tmp_path = tmp.name

        try:
            rings = parse_patterns(tmp_path)
            self.assertEqual(len(rings), 2)
            
            # Ring 1
            r1 = rings[0]
            self.assertEqual(r1["type"], "FAN-OUT")
            self.assertEqual(r1["accounts"], {"001_ACC1", "002_ACC2", "003_ACC3"})
            self.assertEqual(r1["raw_accounts"], {"ACC1", "ACC2", "ACC3"})
            self.assertEqual(r1["first_ts"], pd.Timestamp("2022-09-01 01:00:00"))
            self.assertEqual(r1["last_ts"], pd.Timestamp("2022-09-01 02:00:00"))
            
            # Ring 2
            r2 = rings[1]
            self.assertEqual(r2["type"], "CYCLE")
            self.assertEqual(r2["accounts"], {"001_ACC1", "002_ACC2", "003_ACC3"})
            self.assertEqual(r2["first_ts"], pd.Timestamp("2022-09-01 03:00:00"))
            self.assertEqual(r2["last_ts"], pd.Timestamp("2022-09-01 05:00:00"))
            
            counts = count_rings_by_type(rings)
            self.assertEqual(counts["FAN-OUT"], 1)
            self.assertEqual(counts["CYCLE"], 1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_parse_real_file(self):
        real_path = "data/HI-Small_Patterns.txt"
        if os.path.exists(real_path):
            rings = parse_patterns(real_path)
            self.assertEqual(len(rings), 370)
            counts = count_rings_by_type(rings)
            self.assertEqual(sum(counts.values()), 370)
            for r in rings:
                self.assertIsNotNone(r["type"])
                self.assertGreater(len(r["accounts"]), 0)
                self.assertIsNotNone(r["first_ts"])
                self.assertIsNotNone(r["last_ts"])
                self.assertLessEqual(r["first_ts"], r["last_ts"])

if __name__ == "__main__":
    unittest.main()
