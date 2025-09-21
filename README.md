# Predicting Box Office Revenue Using Weighted Audience Sentiment Analysis

## Overview

This project aims to develop a predictive model for box office revenue using a weighted audience sentiment analysis approach.  The model analyzes movie ratings and reviews from various online platforms, assigning weights based on platform influence and user credibility.  This weighted sentiment is then used as a key feature in predicting a film's financial success, potentially improving investment decisions within the film industry.  The analysis includes data cleaning, feature engineering, model training, and evaluation.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Navigate to the project directory in your terminal and install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print key analysis results to the console, including model performance metrics (e.g., R-squared, RMSE). Additionally, the script generates several visualization plots, including:

* A scatter plot showing the relationship between weighted audience sentiment and box office revenue (`sentiment_vs_revenue.png`).
* A plot illustrating the trend of box office revenue over time (`revenue_trend.png`).  (Other plots may be generated depending on the specific analysis performed).

These plots will be saved in the project's directory.  The specific output and plots may vary depending on the data used and model chosen.