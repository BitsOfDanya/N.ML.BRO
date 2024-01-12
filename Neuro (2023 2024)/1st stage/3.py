n = int(input())
m = int(input())
R = 0
while n != m:
    if n < m:
        n += 3
        R += 1
    if n > m:
        m += 7
        R += 1
print(R)
