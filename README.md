# Learning Analyzer

A Machine Learning minor project (2026) that analyzes student learning behavior and
predicts whether a student will answer a given coding question correctly.

## Overview

Using a dataset of 12,000 question attempts across 150 students, the project explores how
factors like topic, question difficulty, time taken, and number of attempts relate to
answer correctness, then trains and compares several classification models to predict the
outcome (`is_correct`).

## Dataset

`student_learning_dataset.csv` — one row per question attempt, with the following columns:

| Column           | Description                                        |
|------------------|----------------------------------------------------|
| `student_id`     | Student identifier                                 |
| `question_id`    | Question identifier                                |
| `topic`          | Topic (Arrays, Strings, Trees, Graphs, DP, …)      |
| `difficulty`     | 1 = Easy, 2 = Medium, 3 = Hard                     |
| `is_correct`     | Target — 1 if answered correctly, else 0           |
| `time_taken`     | Seconds spent on the attempt                       |
| `attempt_number` | Which attempt this was for the question            |

> Note: the CSV is not committed to this repo. Place `student_learning_dataset.csv` in the
> same folder as the notebook before running.

## Workflow

1. **Exploratory analysis** — accuracy by topic and difficulty, time-taken distributions
   for correct vs. incorrect answers, accuracy vs. attempt number, and student speed vs.
   accuracy.
2. **Preprocessing** — null checks, type coercion, class balancing by downsampling the
   majority class, outlier removal using a z-score threshold on `time_taken`, and one-hot
   encoding of `topic`.
3. **Modeling** — an 80/20 train/test split, then four classifiers compared on accuracy,
   precision, recall, and F1.

## Results

| Model                | Accuracy | F1     |
|----------------------|----------|--------|
| Logistic Regression  | 80.37%   | 0.815  |
| Random Forest        | 80.14%   | 0.816  |
| Decision Tree        | 79.68%   | 0.822  |
| AdaBoost             | 77.60%   | 0.774  |

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter Notebook

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook ML_PROJECT_v1.ipynb
```

Then run the cells top to bottom (make sure the dataset CSV is in the same folder).
