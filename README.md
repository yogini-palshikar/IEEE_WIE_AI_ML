# Iris Flower Classification: Decision Tree vs. KNN

A beginner-friendly machine learning project that predicts the **species of an iris flower** from four measurements, and compares two simple models: a **Decision Tree** and **K-Nearest Neighbors (KNN)**.

## Project overview

- **Goal:** predict whether an iris is *setosa*, *versicolor* or *virginica* from its sepal length, sepal width, petal length and petal width (cm).
- **Why it matters:** it is a small, real classification task that shows the complete workflow (load, check, prepare, train, evaluate, compare) and applies to many real problems, from species identification to medical screening.
- **Steps in the notebook:** data loading and missing-value check → features/target split → two models → accuracy + confusion matrices → 5-fold cross-validation → conclusion.

## Dataset

- **Iris dataset** (R. A. Fisher, 1936): 150 flowers, 50 per species, 4 numeric features, no missing values.
- Link: <https://archive.ics.uci.edu/dataset/53/iris>
- The notebook loads it straight from scikit-learn (`sklearn.datasets.load_iris`), so there is nothing to download.

## Setup instructions

```bash
# 1. Clone this repository
git clone https://github.com/yogini-palshikar/IEEE_WIE_AI_ML.git
cd IEEE_WIE_AI_ML

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook iris_classification.ipynb
```

Then choose **Kernel → Restart & Run All**. It also runs in Google Colab: upload the notebook and run it, since the libraries are preinstalled.

## Results

Train/test split: 80/20 (120 train, 30 test), stratified, `random_state=42`.

| Model | Test accuracy (30 flowers) | 5-fold CV accuracy (150 flowers) |
|---|---|---|
| Decision Tree (`max_depth=3`) | **96.7%** (29/30) | 96.0% ± 2.5% |
| KNN (`k=5`, standardized features) | 93.3% (28/30) | **97.3%** ± 2.5% |

The confusion matrices and decision-tree plot are in the notebook. The Decision Tree made 1 error (a versicolor predicted as virginica). KNN made 2 errors (virginica predicted as versicolor). Setosa was classified perfectly by both.

## Conclusion

- On the held-out test set the **Decision Tree was better** (96.7% vs. 93.3%, a difference of one flower). Under 5-fold cross-validation **KNN was slightly ahead** (97.3% vs. 96.0%).
- The differences are smaller than the fold-to-fold variation, so the two models are **effectively tied** on this dataset.
- **What I learned:** a single small test split can change the ranking, so cross-validation gives a fairer comparison; both models only confuse versicolor and virginica; the tree is easy to explain because just petal length and petal width drive its decisions; and KNN needs feature scaling while trees do not.
- **Next steps:** tune `k` and `max_depth` with grid search, try a Random Forest, and repeat on a harder dataset such as mushroom classification.

## Repository contents

```
├── iris_classification.ipynb   # the full notebook (outputs included)
├── README.md
└── requirements.txt
```
