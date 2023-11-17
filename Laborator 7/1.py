import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('auto-mpg.csv')

print(df.isnull().sum())

df_cleaned = df.dropna()

plt.scatter(df_cleaned['horsepower'], df_cleaned['mpg'])
plt.xlabel('Cai putere')
plt.ylabel('mpg ')
plt.title('Relația dintre cp și mpg')
plt.show()
