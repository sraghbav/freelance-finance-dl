# Freelance Finance Deep Learning Project

## Project Overview

This project investigates whether deep learning models can learn financial behavior patterns from transaction histories. The goal is to model personal finance data as temporal sequences and predict future transaction behavior.

Rather than treating each transaction as an independent row, transactions are grouped into sequences that capture spending and income patterns over time. This structure is intended for future sequence-based models such as LSTMs.

---

## Dataset

This project uses the BudgetWise Personal Finance Dataset.

The dataset contains transaction-level information including:

- User ID
- Date
- Transaction Type
- Category
- Payment Mode
- Amount
- Location
- Notes

Dataset source:

https://www.kaggle.com/datasets

(Replace with the exact BudgetWise dataset URL.)

---

## Repository Structure

```text
freelance_finance_dl/
│
├── freelance_finance_dl/
│   ├── __init__.py
│   └── dataloader.py
│
├── data/
│   └── budgetwise_finance_dataset.csv
│
├── notebooks/
│   └── data_demo.ipynb
│
├── setup.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-github-url>
cd freelance_finance_dl
```

Install the package:

```bash
pip install -e .
```

Verify installation:

```python
from freelance_finance_dl.dataloader import FinanceTransactionDataset
```

---

## Data Loader

The data loader creates transaction sequences for deep learning models.

Each sample contains:

```text
Previous 5 transactions -> Next transaction amount
```

Current feature set:

1. Transaction Type
2. Category
3. Payment Mode
4. Amount

Example shape:

```python
features.shape
```

Output:

```text
torch.Size([5, 4])
```

Meaning:

```text
5 transactions
4 features per transaction
```

Target:

```python
target
```

Represents:

```text
Next transaction amount
```

---

## Example Usage

```python
from freelance_finance_dl.dataloader import FinanceTransactionDataset

dataset = FinanceTransactionDataset(
    "data/budgetwise_finance_dataset.csv",
    sequence_length=5
)

features, target = dataset[0]

print(features.shape)
print(target)
```

---

## Running the Demo Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/data_demo.ipynb
```

The notebook demonstrates:

- Loading the raw dataset
- Creating transaction sequences
- Inspecting sample inputs
- Creating PyTorch DataLoaders
- Verifying sequence shapes

---

## Future Work

Planned next steps include:

- LSTM-based sequence modeling
- Transaction forecasting
- Financial anomaly detection
- Comparison against baseline models
- Evaluation using MAE and MSE

---

## Author

Sriraghav Bavineni

DSCI 410L – Introduction to Deep Learning