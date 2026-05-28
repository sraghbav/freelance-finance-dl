# Freelance Finance Deep Learning Project

## Project Overview

This project investigates whether deep learning models can learn financial behavior patterns from personal finance transaction histories.

The project uses transaction-level finance data and converts individual transactions into temporal sequences. Instead of treating each transaction as an independent row, the dataloader groups previous transactions together so the model can eventually learn patterns over time.

Current prediction setup:

```text
previous 5 transactions → next transaction amount
```

This sequence-based structure is intended for future deep learning models such as LSTMs or other sequence models.

---

## Dataset

This project uses the BudgetWise Personal Finance Dataset from Kaggle:

https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset

The dataset contains personal finance transaction data, including:

- user ID
- date
- transaction type
- category
- amount
- payment mode
- location
- notes

For this milestone, the dataset file is included in the `data/` folder.

---

## Repository Link

https://github.com/sraghbav/freelance-finance-dl.git

---

## Repository Structure

```text
freelance-finance-dl/
├── freelance_finance_dl/
│   ├── __init__.py
│   └── dataloader.py
├── data/
│   └── budgetwise_finance_dataset.csv
├── notebooks/
│   └── data_demo.ipynb
├── setup.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sraghbav/freelance-finance-dl.git
cd freelance-finance-dl
```

Install the package:

```bash
pip install -e .
```

After installation, the dataloader can be imported with:

```python
from freelance_finance_dl.dataloader import FinanceTransactionDataset
```

---

## Data Loader

The dataloader creates sequence samples from each user's transaction history.

Each sample contains:

```text
previous sequence_length transactions → next transaction amount
```

By default:

```text
sequence_length = 5
```

Each transaction has 4 features:

1. encoded transaction type
2. encoded category
3. encoded payment mode
4. amount

Example input shape for one sample:

```text
torch.Size([5, 4])
```

This means:

```text
5 previous transactions
4 features per transaction
```

The target is a single value:

```text
next transaction amount
```

---

## Example Usage

```python
from freelance_finance_dl.dataloader import FinanceTransactionDataset, get_data_loader

dataset = FinanceTransactionDataset(
    csv_file="data/budgetwise_finance_dataset.csv",
    sequence_length=5
)

features, target = dataset[0]

print(features.shape)
print(target)
```

Expected feature shape:

```text
torch.Size([5, 4])
```

A PyTorch DataLoader can also be created:

```python
loader = get_data_loader(
    csv_file="data/budgetwise_finance_dataset.csv",
    batch_size=32,
    sequence_length=5,
    shuffle=True
)

batch_features, batch_targets = next(iter(loader))

print(batch_features.shape)
print(batch_targets.shape)
```

Expected batch shapes:

```text
torch.Size([32, 5, 4])
torch.Size([32])
```

---

## Demo Notebook

The notebook is located at:

```text
notebooks/data_demo.ipynb
```

The notebook demonstrates:

- loading the raw dataset
- importing the installed package
- creating transaction sequences
- inspecting one sample
- creating a PyTorch DataLoader
- verifying that the dataloader returns sequence-shaped tensors

Run the notebook with:

```bash
jupyter notebook notebooks/data_demo.ipynb
```

---

## Current Status

This milestone includes:

- installable Python package
- sequence-based PyTorch dataset
- PyTorch DataLoader helper function
- runnable data demo notebook
- dataset included in GitHub repository

---

## Future Work

Next steps include:

- building an LSTM or other sequence model
- comparing against a simple baseline model
- evaluating predictions using MAE and MSE
- exploring anomaly detection for transactions that differ strongly from predicted behavior

---

## Author

Sriraghav Bavineni  
DSCI 410L