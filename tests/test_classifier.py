
import unittest
import pandas as pd
import numpy as np
from ml_auto_classifier import ClassificationEvaluator

class TestClassificationEvaluator(unittest.TestCase):
    def setUp(self):
        # Create simple dataset for testing
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        self.y = pd.Series(np.random.randint(0, 2, 100))
        self.evaluator = ClassificationEvaluator()

    def test_evaluate_models(self):
        results = self.evaluator.evaluate_models(self.X, self.y)
        self.assertIn('model_results', results)
        self.assertIn('dataset_info', results)

    def test_llm_analysis(self):
        self.evaluator.evaluate_models(self.X, self.y)
        analysis = self.evaluator.get_llm_analysis()
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 0)

if __name__ == '__main__':
    unittest.main()