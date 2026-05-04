import pandas as pd

# Load dataset
df = pd.read_csv("data/steel_industry_data.csv")

# Show basic info
print(df.head())
print(df.info())

# Handle missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)

print("Data preprocessing completed!")