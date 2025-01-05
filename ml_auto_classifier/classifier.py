import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from typing import Union, Dict, Optional
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os

class ClassificationEvaluator:
    """A class to evaluate multiple classification models with optional dataset reduction."""
    
    def __init__(self, api_key: Optional[str] = None, random_state: int = 42):
        """
        Initialize the ClassificationEvaluator.
        
        Args:
            api_key (Optional[str]): Google API key. If None, will try to get from environment variable
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = self._setup_logger()
        self.results = None
        
        # Configure Gemini API with flexible key handling
        self._setup_api(api_key)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ClassificationEvaluator')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def _setup_api(self, api_key: Optional[str] = None):
        """
        Set up the Gemini API with flexible key handling.
        
        Args:
            api_key (Optional[str]): Google API key
        """
        try:
            # Try loading from provided key first
            if api_key:
                genai.configure(api_key=api_key)
            else:
                # Try loading from environment
                load_dotenv()  # This will load from .env file if present
                env_key = os.getenv('GOOGLE_API_KEY')
                if env_key:
                    genai.configure(api_key=env_key)
                else:
                    self.logger.warning("No API key provided. LLM analysis will not be available.")
                    return
                
            self.model = genai.GenerativeModel('gemini-pro')
            
        except Exception as e:
            self.logger.error(f"Error setting up Gemini API: {str(e)}")
            self.model = None

    def set_api_key(self, api_key: str):
        """
        Set or update the API key.
        
        Args:
            api_key (str): Google API key
        """
        self._setup_api(api_key)
    
    def _reduce_dataset(self, X: pd.DataFrame, y: pd.Series, reduction_percent: float) -> tuple:
        """
        Reduce dataset size by given percentage.
        
        Args:
            X: Feature matrix
            y: Target variable
            reduction_percent: Percentage of data to keep (0.0 to 1.0)
            
        Returns:
            tuple: Reduced X and y
        """
        if reduction_percent < 1.0:
            sample_size = int(len(X) * reduction_percent)
            indices = np.random.choice(len(X), sample_size, replace=False)
            return X.iloc[indices], y.iloc[indices]
        return X, y
    
    def evaluate_models(self, 
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       dataset_fraction: float = 0.4,
                       test_split: float = 0.6) -> Dict:
        """
        Evaluate multiple classification models on the dataset.
        
        Args:
            X: Feature matrix
            y: Target variable
            dataset_fraction: Fraction of dataset to use (default: 0.4)
            test_split: Test set proportion for model evaluation
            
        Returns:
            Dict: Dictionary containing evaluation results
        """
        try:
            # Convert numpy arrays to pandas if necessary
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            
            original_size = len(X)
            
            # Reduce dataset if specified
            X, y = self._reduce_dataset(X, y, dataset_fraction)
            
            self.logger.info(f"Original dataset size: {original_size}")
            self.logger.info(f"Reduced dataset size: {len(X)} ({dataset_fraction*100}% of original)")
            
            # Split the reduced dataset for training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=self.random_state
            )
            
            # Initialize and fit lazy classifier
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            self.results = models  
            
            return {
                'dataset_info': {
                    'original_size': original_size,
                    'reduced_size': len(X),
                    'reduction_percentage': dataset_fraction * 100
                },
                'model_results': self.results
            }
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def get_llm_analysis(self) -> str:
        """
        Use Google's Gemini to analyze model performances and provide recommendations.
        
        Returns:
            str: LLM analysis of model performance
        """
        if self.results is None:
            return "No models have been evaluated yet. Please run evaluate_models first."
            
        if self.model is None:
            return "LLM analysis not available. Please set a valid API key using set_api_key() method."
        
        # Prepare complete model performance data for LLM
        performance_text = "Complete Model Performance Results:\n"
        for idx, row in self.results.iterrows():
            performance_text += f"\nModel: {idx}\n"
            for metric, value in row.items():
                if isinstance(value, (int, float)):
                    performance_text += f"{metric}: {value:.4f}\n"
                else:
                    performance_text += f"{metric}: {value}\n"
        
        prompt = f"""As an expert machine learning engineer, identify the top 5 best-performing models from the provided results. Prioritize models based on the following criteria:

1. Accuracy
2. F1-score
3. Precision
4. Recall
5. Training/Prediction Time (lower is better in case of a tie)

Here are the model performance results:

{performance_text}

Provide the names of the top 5 models along with their key metrics, and at the end give the parameters of only first top performing model for Grid SearchCV in proper format so that user can directly use it in code and if parameter not avilable then not need to write anything ."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error getting LLM analysis: {str(e)}")
            return f"Error getting LLM analysis: {str(e)}"

def quick_classify(X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  dataset_fraction: float = 0.8,
                  test_split: float = 0.2) -> Dict:
    """
    Convenience function to evaluate models and get LLM analysis.
    
    Args:
        X: Feature matrix
        y: Target variable
        dataset_fraction: Fraction of dataset to use (default: 0.8)
        test_split: Test set proportion for model evaluation
        
    Returns:
        Dict: Results and LLM analysis
    """
    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_models(X, y, dataset_fraction, test_split)
    results['llm_analysis'] = evaluator.get_llm_analysis()
    return results