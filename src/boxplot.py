from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Boxplot: 
plt.figure()
BP = df[["MedHouseVal"]].boxplot()

BP.set_xlabel("")
BP.set_xticks([])

plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")
plt.xlabel("")

# Save figure
plt.savefig("figs/boxplot.png")
plt.close()

