import pandas as pd
file_path = 'init_data.csv'
df = pd.read_csv(file_path, delimiter=';')

sum_squares_rows = df.apply(lambda row: ((row - row.mean())**2).sum() / len(df.columns), axis=1)
sum_squares_columns = df.apply(lambda col: ((col - col.mean())**2).sum() / len(df), axis=0)

max_row_index = sum_squares_rows.idxmax()
min_column_index = sum_squares_columns.idxmin()

min_column_index_num = int(min_column_index) if isinstance(min_column_index, str) else min_column_index
index_sum = max_row_index + min_column_index_num
even_rows = df.iloc[::2]
odd_columns = even_rows.iloc[:, 1::2]

sum_result = odd_columns.sum().sum()
rounded_sum = round(sum_result, 1)
print(rounded_sum, max_row_index, min_column_index, index_sum)
