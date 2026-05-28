# Freelance Finance Deep Learning Project

## Project Overview

This project investigates how deep learning can be used to identify unusual financial behavior in personal finance transaction data.

The project uses category-specific transaction sequences and a sequence autoencoder to learn normal spending patterns. Rather than forecasting future transactions, the model is trained to reconstruct transaction sequences. Sequences that cannot be reconstructed well are considered potential anomalies.

This approach is designed to identify unusual financial behavior such as spending spikes, irregular transactions, or other patterns that differ significantly from a user's typical financial activity.

Current project pipeline:

```text
Transactions
    ↓
Group by User and Category
    ↓
Create Transaction Sequences
    ↓
Sequence Autoencoder
    ↓
Reconstruction Error
    ↓
Anomaly Detection
```

---

## Dataset

This project uses the BudgetWise Personal Finance Dataset from Kaggle:

https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset

The dataset contains personal finance transaction data including:

- User ID
- Date
- Transaction Type
- Category
- Amount
- Payment Mode
- Location
- Notes

The dataset is included in the repository under the `data/` directory.

---

## Repository

GitHub Repository:

https://github.com/sraghbav/freelance-finance-dl.git

---

## Repository Structure

```text
freelance-finance-dl/
│
├── freelance_finance_dl/
│   ├── __init__.py
│   ├── dataloader.py
│   └── model.py
│
├── data/
│   └── budgetwise_finance_dataset.csv
│
├── notebooks/
│   └── data_demo.ipynb
│
├── README.md
├── requirements.txt
└── setup.py
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

Verify installation:

```python
from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.model import TransactionAutoencoder
```

---

## Data Loader

The dataloader creates category-specific transaction sequences.

Transactions are grouped by:

```text
user_id
+
category
```

Example transaction history:

```text
Food:
18
22
20
25
19
24
```

Generated sequence:

```text
[18, 22, 20, 25, 19]
```

The autoencoder uses the same sequence as both the input and target:

```text
Input:
[18, 22, 20, 25, 19]

Target:
[18, 22, 20, 25, 19]
```

This allows the model to learn normal spending behavior.

Example sample shape:

```text
torch.Size([5, 1])
```

Example batch shape:

```text
torch.Size([32, 5, 1])
```

---

## Autoencoder Model

The project uses a sequence autoencoder.

The encoder compresses transaction sequences into a lower-dimensional representation.

The decoder reconstructs the original transaction sequence.

Pipeline:

```text
Transaction Sequence
        ↓
Encoder
        ↓
Latent Representation
        ↓
Decoder
        ↓
Reconstructed Sequence
```

The reconstruction error is then used as an anomaly score.

---

## Anomaly Detection

After reconstruction:

```text
Original Sequence
vs
Reconstructed Sequence
```

The reconstruction error is calculated using Mean Squared Error (MSE).

Example:

```text
Original:
[18, 22, 20, 25, 19]

Reconstructed:
[18, 21, 20, 24, 20]
```

Small reconstruction error:

```text
Normal behavior
```

Example:

```text
Original:
[18, 22, 20, 25, 300]

Reconstructed:
[18, 21, 20, 24, 22]
```

Large reconstruction error:

```text
Potential anomaly
```

Anomaly threshold:

```text
mean reconstruction error
+
2 × standard deviations
```

---

## Example Usage

### Create Dataset

```python
from freelance_finance_dl.dataloader import FinanceTransactionDataset

dataset = FinanceTransactionDataset(
    csv_file="data/budgetwise_finance_dataset.csv",
    sequence_length=5
)

print(len(dataset))
```

### Create DataLoader

```python
from freelance_finance_dl.dataloader import get_data_loader

loader = get_data_loader(
    csv_file="data/budgetwise_finance_dataset.csv",
    batch_size=32,
    sequence_length=5
)

batch_x, batch_y = next(iter(loader))

print(batch_x.shape)
print(batch_y.shape)
```

Expected output:

```text
torch.Size([32, 5, 1])
torch.Size([32, 5, 1])
```

### Load Autoencoder

```python
from freelance_finance_dl.model import TransactionAutoencoder

model = TransactionAutoencoder(
    sequence_length=5
)

reconstructed = model(batch_x)

print(reconstructed.shape)
```

Expected output:

```text
torch.Size([32, 5, 1])
```

---

## Demo Notebook

The demonstration notebook is located at:

```text
notebooks/data_demo.ipynb
```

The notebook demonstrates:

- Loading the raw dataset
- Creating category-specific sequences
- Loading data using the package
- Creating a PyTorch DataLoader
- Running sequences through the autoencoder
- Calculating reconstruction loss
- Calculating anomaly scores
- Identifying potential anomalies

---

## Evaluation

The project evaluates anomaly detection using reconstruction error.

Metrics:

- Mean Squared Error (MSE)
- Reconstruction Error Distribution
- Number of Flagged Anomalies

The model's goal is to learn normal transaction behavior and identify sequences that differ significantly from expected patterns.

---

## Future Thesis Extension

This project serves as the deep learning foundation for a larger honors thesis project.

Future extensions include:

### LLM Financial Guidance

```text
Transaction Sequences
        ↓
Autoencoder
        ↓
Anomaly Detection
        ↓
LLM Explanation Layer
```

Example:

```text
This transaction is significantly larger than your typical spending within this category and may warrant review.
```

### Tax Planning Assistance

Future versions may provide:

- Estimated tax set-aside recommendations
- Potential business expense identification
- Expense documentation reminders
- Financial pattern summaries

Example:

```text
Based on your freelance income and recorded expenses, consider setting aside approximately 25–30% of net income for tax obligations.
```

This guidance would be educational only and not professional tax advice.

---

## Author

Sriraghav Bavineni

DSCI 410L Deep Learning Project

University of Oregon