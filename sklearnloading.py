from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("Iris dataset:")
print(df.head())

diabetes= datasets.load_diabetes()
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target
print("\nDiabetes dataset:")
print(diabetes_df.head())