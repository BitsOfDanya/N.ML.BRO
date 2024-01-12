count = 0

for i in range(2, 65):  # начинаем с 2, так как x должен быть больше y
    for j in range(1, i):  # j < i чтобы удовлетворять условию x > y
        x = 2 ** i - 1
        y = 2 ** j - 1
        product = x * y

        binary_str = bin(product)[2:]
        ones = binary_str.count("1")
        zeros = binary_str.count("0")

        if abs(ones - zeros) <= 13:
            count += 1

print("Количество пар (x, y):", count)
