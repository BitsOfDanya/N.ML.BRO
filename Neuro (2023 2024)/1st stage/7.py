n = int(input())
cell_sizes = list(map(int, input().split()))

time_counter = 0  # Счетчик времени
current_index = 0  # Индекс текущей ячейки

cell_index_map = {size: index for index, size in enumerate(cell_sizes)}

for detail_size in range(1, n + 1):
    target_index = cell_index_map[detail_size]

    time_to_target = (target_index - current_index) % n

    time_counter += time_to_target

    current_index = target_index
print(time_counter)




