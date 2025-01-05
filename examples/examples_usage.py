import pandas as pd
from sklearn.datasets import load_iris
from ml_auto_classifier import ClassificationEvaluator

# Load example dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Initialize evaluator
evaluator = ClassificationEvaluator()

# Evaluate models
results = evaluator.evaluate_models(X, y, dataset_fraction=0.8)

# Get AI analysis
analysis = evaluator.get_llm_analysis()

# Print results
print("Model Results:")
print(results['model_results'])
print("\nAI Analysis:")
print(analysis)