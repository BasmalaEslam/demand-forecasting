import os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['ENABLE_LLM']='0'
import app, common

import numpy as np, pandas as pd
class DemandTests(unittest.TestCase):
    def test_features_do_not_use_current_or_future_target(self):
        data=app.read_data();index=100
        original=app.features(data).iloc[index]
        changed=data.copy();changed.loc[index:,'units']=999999
        np.testing.assert_allclose(original,app.features(changed).iloc[index])
    def test_forecast_dates_and_reorder(self):
        r=app.dispatch('/forecast',{'days':10,'lead_time_days':7,'stock_on_hand':0})
        self.assertEqual(len(r['forecast']),10)
        self.assertGreater(r['planning']['recommended_order_units'],0)
        self.assertTrue(all(row['rough_lower']<=row['units']<=row['rough_upper'] for row in r['forecast']))
        self.assertEqual(pd.Timestamp(r['forecast'][0]['date']),app.read_data().date.iloc[-1]+pd.Timedelta(days=1))
    def test_large_stock_needs_no_order(self):
        self.assertEqual(app.dispatch('/forecast',{'days':7,'lead_time_days':7,'stock_on_hand':100000})['planning']['recommended_order_units'],0)
    def test_invalid_horizons_rejected(self):
        for payload in [{'days':0},{'days':61},{'days':True},{'days':3,'lead_time_days':4},{'days':7,'stock_on_hand':float('inf')}]:
            with self.subTest(payload=payload),self.assertRaises(ValueError):app.dispatch('/forecast',payload)
    def test_one_step_and_recursive_evaluation_both_present(self):
        r=app.dispatch('/report',{})
        self.assertIn('test_last_60_days_recursive',r)
        self.assertIn('seasonal_naive_lag7_one_step',r)
if __name__=='__main__':unittest.main()
