import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 500000

# Generate realistic features
square_feet = np.random.normal(2000, 500, n).clip(800, 5000)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.15, 0.4, 0.25, 0.1, 0.05])
bathrooms = (bedrooms * 0.75 + np.random.normal(0, 0.3, n)).clip(1, 6).round(1)
lot_size_sqft = (square_feet * np.random.uniform(2, 4, n) + np.random.normal(0, 1000, n)).clip(2000, 50000)
current_year = 2025
year_built = np.random.normal(1990, 15, n).astype(int).clip(1900, current_year)
distance_to_city_center = np.random.exponential(8, n).clip(0.1, 30).round(1)
neighborhood_rating = np.random.normal(7, 1, n).clip(1, 10).round(1)
avg_income_area = np.random.normal(70000, 20000, n).clip(30000, 150000).round(-3)
crime_rate = (15 - neighborhood_rating + np.random.normal(0, 1, n)).clip(0.1, 30).round(1)

# Calculate price based on features with some randomness
base_price = (
    200 * square_feet +
    50000 * bedrooms +
    40000 * bathrooms +
    2 * lot_size_sqft +
    1000 * (current_year - year_built) * -1 +
    -10000 * distance_to_city_center +
    30000 * neighborhood_rating +
    0.5 * avg_income_area +
    -5000 * crime_rate
)
price = (base_price + np.random.normal(0, 50000, n)).clip(100000, 2000000).round(-3)

# Create DataFrame
df = pd.DataFrame({
    'square_feet': square_feet.round().astype(int),
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'lot_size_sqft': lot_size_sqft.round().astype(int),
    'year_built': year_built,
    'distance_to_city_center': distance_to_city_center,
    'neighborhood_rating': neighborhood_rating,
    'avg_income_area': avg_income_area.astype(int),
    'crime_rate': crime_rate,
    'price': price.astype(int)
})

# Save to CSV
df.to_csv('real_estate_data.csv', index=False)
print('Dataset generated successfully!')