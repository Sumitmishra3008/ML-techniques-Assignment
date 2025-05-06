import pandas as pd
data={
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
    'Salary': [70000, 80000, 90000, 100000]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
df.to_csv('sample.csv', index=False)
print("\nDataFrame saved to 'sample.csv'")
