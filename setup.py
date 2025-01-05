from setuptools import setup, find_packages

setup(
    name="ml_auto_classifier",
    version="0.1.0",
    author="Alok Kumar",
    description="An automatic machine learning classification evaluation tool",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lazypredict",
        "google-generativeai",
        "python-dotenv"
    ],
    python_requires=">=3.6"
)