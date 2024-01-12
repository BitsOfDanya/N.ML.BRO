n = int(input().strip())
transactions = [list(map(int, input().strip().split())) for _ in range(n)]

unique_units = {}

total_unique_units = 0

for frm, to, amount in transactions:
    if frm not in unique_units:
        unique_units[frm] = 0
    if unique_units[frm] < amount:
        total_unique_units += amount - unique_units[frm]
        unique_units[frm] = 0
    else:
        unique_units[frm] -= amount

    if to not in unique_units:
        unique_units[to] = 0
    unique_units[to] += amount

print(total_unique_units)
