from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-auto-classifier",
    version="0.1.0",
    author="Alok Kumar",
    author_email="ay747283@gmail.com",
    description="Automated machine learning classification with LLM analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alok-Kumar2005/ML-Library",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "lazypredict>=0.2.12",
        "groq>=0.3.0",
        "python-dotenv"
    ],
)