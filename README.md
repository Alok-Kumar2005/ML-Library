# This is a ML Library Code

```
conda create -p venv python=3.9 -y
```

```
conda activate venv/
```

```
pip install -r requriements.txt
```

```
how to use the code  ( Apply after Preprocessing and Feature Engineering )
1. !git clone https://github.com/Alok-Kumar2005/ML-Library
2. %cd ML-Library/
3. !pip install -r requirements.txt
 
from ml_auto_classifier import ClassificationEvaluator

evaluator = ClassificationEvaluator(api_key = "Your Google API key")

results = evaluator.evaluate_models(X, y, dataset_fraction=0.4)   ## X: dataset , y: target column , dataset_fraction: on how much dataset you want to perform operation

analysis = evaluator.get_llm_analysis()
print(analysis)

```