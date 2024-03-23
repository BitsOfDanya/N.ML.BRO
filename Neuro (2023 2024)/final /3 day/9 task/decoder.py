import csv

file_path = 'eeg_data.csv'
column_index = 3
column_values = []

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if len(row) > column_index:
            column_values.append(row[column_index])

for k in column_values:
    separator = "\\"
    A0 = []
    A1 = []
    A2 = []
    A3 = []
    A4 = []
    A5 = []
    A6 = []
    A7 = []
    A8 = []
    A9 = []
    result_list = k.split(separator)
    # print(result_list)
    for i in result_list:
        if i[-1:] == '0':
            A0.append(i)
        if i[-1:] == '1':
            A1.append(i)
        if i[-1:] == '2':
            A2.append(i)
        if i[-1:] == '3':
            A3.append(i)
        if i[-1:] == '4':
            A4.append(i)
        if i[-1:] == '5':
            A5.append(i)
        if i[-1:] == '6':
            A6.append(i)
        if i[-1:] == '7':
            A7.append(i)
        if i[-1:] == '8':
            A8.append(i)
        if i[-1:] == '9':
            A9.append(i)

    max_len = max(len(A0), len(A1), len(A2), len(A3), len(A4), len(A5), len(A6), len(A7), len(A8), len(A9))

    A0 += [''] * (max_len - len(A0))
    A1 += [''] * (max_len - len(A1))
    A2 += [''] * (max_len - len(A2))
    A3 += [''] * (max_len - len(A3))
    A4 += [''] * (max_len - len(A4))
    A5 += [''] * (max_len - len(A5))
    A6 += [''] * (max_len - len(A6))
    A7 += [''] * (max_len - len(A7))
    A8 += [''] * (max_len - len(A8))
    A9 += [''] * (max_len - len(A9))

with open('output.csv', 'w') as file:
    for i in range(max_len):
        line = f"{A0[i]},{A1[i]},{A2[i]},{A3[i]},{A4[i]},{A5[i]},{A6[i]},{A7[i]},{A8[i]},{A9[i]}\n"
        file.write(line)



