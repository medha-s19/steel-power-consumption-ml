import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("START")

df = pd.read_csv("data/Steel_industry_data.csv")

print("Shape:", df.shape)

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary stats:")
print(df.describe())

plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(x=df["Usage_kWh"])
plt.show()


target = "Usage_kWh"

if target in df.columns:
    plt.figure()
    df[target].hist(bins=30)
    plt.title("Target Distribution")
    plt.show()
