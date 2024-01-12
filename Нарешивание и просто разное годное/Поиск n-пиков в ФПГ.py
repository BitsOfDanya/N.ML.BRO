def find_peaks(signal):
    peaks = 0
    min_height = 100
    min_distance = 20
    min_val = signal[0]
    last_peak = -min_distance

    for i in range(1, len(signal)):
        if signal[i] > signal[i-1]:
            if signal[i] - min_val > min_height and i - last_peak > min_distance:
                peaks += 1
                last_peak = i
            min_val = min(min_val, signal[i-1])
        else:
            min_val = min(min_val, signal[i])

    return peaks

def main():
    data = input("Введите данные: ")
    data = [int(x) for x in data.split(',')]
    print(find_peaks(data))

if __name__ == "__main__":
    main()
