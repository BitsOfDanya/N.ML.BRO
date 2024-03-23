from psychopy import visual, core, event
import random
import serial
import csv
from time import sleep, time
import threading
import pandas as pd
import matplotlib.pyplot as plt

square_state = 0
square_change_count = 0

def collect_data_from_arduino(arduino, file_name='arduino_data.csv'):
    global square_state
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Data', 'SquareState'])

        start_time = time()
        while not event.getKeys(keyList=['escape']):
            if arduino.inWaiting() > 0:
                data_line = arduino.readline().decode('utf-8').strip()
                if data_line.startswith('R') or data_line.startswith('C'):
                    flashing_info = data_line
                    writer.writerow([time() - start_time, 'Flashing', flashing_info])
                else:
                    data_values = data_line.split(',')
                    if len(data_values) == 10:
                        writer.writerow([time() - start_time] + data_values + [square_state])

def start_arduino_data_collection(port='COM3', baud_rate=115200, file_name='arduino_data.csv'):
    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        sleep(2)
        data_collection_thread = threading.Thread(target=collect_data_from_arduino, args=(arduino, file_name))
        data_collection_thread.start()
        return data_collection_thread
    except Exception as e:
        print(f"Could not connect to Arduino: {e}")
        exit()

def run_psychopy_tasks():
    global square_state, square_change_count
    win = visual.Window(size=(800, 700), color='white', units='pix')
    alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,_'
    symbols = [visual.TextStim(win, text=char, height=30, color='black', pos=(i % 6 * 100 - 250, 200 - (i // 6) * 100)) for i, char in enumerate(alphabet)]
    black_square = visual.Rect(win, width=50, height=50, fillColor='black', lineColor='black', pos=(-350, -250))

    change_started = False
    change_time = 0.75
    last_change_time = core.getTime()

    while not event.getKeys(keyList=['escape']):
        for symbol in symbols:
            symbol.draw()
        black_square.draw()
        win.flip()

        keys = event.getKeys()
        if 'space' in keys:
            change_started = True

        if change_started and core.getTime() - last_change_time >= change_time:
            random_symbol = random.choice(symbols)
            random_symbol.color = 'white' if random_symbol.color == 'black' else 'black'

            if square_state == 0:
                black_square.fillColor = 'white'
                square_state = 1
                square_change_count += 1
            else:
                black_square.fillColor = 'black'
                square_state = 0

            last_change_time = core.getTime()

    win.close()

if __name__ == "__main__":
    data_collection_thread = start_arduino_data_collection(port='COM3', baud_rate=115200, file_name='arduino_data.csv')
    run_psychopy_tasks()

    if data_collection_thread.is_alive():
        data_collection_thread.join()

    print(f"Количество изменений состояния квадрата: {square_change_count}")

    data = pd.read_csv('arduino_data.csv')
    plt.plot(data['Timestamp'], data['SquareState'], label='Состояние квадрата')
    plt.xlabel('Время')
    plt.ylabel('Состояние')
    plt.title('Изменения состояния квадрата')
    plt.legend()
    plt.show()
