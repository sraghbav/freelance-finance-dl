from setuptools import setup, find_packages

setup(
    name="freelance_finance_dl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "jupyter",
    ],
)
