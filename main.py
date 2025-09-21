import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'Budget': np.random.randint(1000000, 100000000, num_movies),
    'RottenTomatoes': np.random.randint(0, 101, num_movies),
    'IMDB': np.random.uniform(0, 10, num_movies),
    'Metacritic': np.random.randint(0, 101, num_movies),
    'AudienceScore': np.random.randint(0, 101, num_movies),
    'BoxOfficeRevenue': np.random.randint(100000, 1000000000, num_movies)
}
df = pd.DataFrame(data)
# Weighting Audience Sentiment (example weighting)
df['WeightedSentiment'] = (df['RottenTomatoes'] * 0.3 + 
                           df['IMDB'] * 0.3 + 
                           df['Metacritic'] * 0.2 + 
                           df['AudienceScore'] * 0.2)
# --- 2. Data Cleaning (example - handling outliers) ---
# (Illustrative -  Robust outlier removal could be more sophisticated)
df = df[df['BoxOfficeRevenue'] < df['BoxOfficeRevenue'].quantile(0.95)]
# --- 3. Analysis ---
# Linear Regression to predict Box Office Revenue based on Weighted Sentiment
slope, intercept, r_value, p_value, std_err = linregress(df['WeightedSentiment'], df['BoxOfficeRevenue'])
print("Linear Regression Results:")
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='WeightedSentiment', y='BoxOfficeRevenue', data=df)
plt.title('Box Office Revenue vs. Weighted Audience Sentiment')
plt.xlabel('Weighted Audience Sentiment')
plt.ylabel('Box Office Revenue')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'box_office_sentiment.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve more sophisticated modeling techniques (e.g., RandomForest, Gradient Boosting) and feature engineering.