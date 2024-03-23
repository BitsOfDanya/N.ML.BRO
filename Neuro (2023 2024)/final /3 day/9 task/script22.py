import psychopy.visual
import psychopy.event
import random
import time
import serial
import csv

win = psychopy.visual.Window(fullscr=True, units='pix', color='gray')

font = 'Arial'

alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,_'
table_size = 6
table = [alphabet[i:i + table_size] for i in range(0, len(alphabet), table_size)]
table = [line.ljust(table_size, ' ') for line in table]

square_size = 50
sync_square = psychopy.visual.Rect(win, width=square_size, height=square_size, fillColor='black', pos=[300, -300])

text_objects = []
for i in range(table_size):
    for j in range(table_size):
        text = psychopy.visual.TextStim(win, text=table[i][j], font=font, height=30,
                                        pos=(-300 + j * 100, 200 - i * 100))
        text_objects.append(text)

arduino_port = 'COM3'
arduino_baudrate = 115200
arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=1)

with open('eeg_data.csv', 'w', newline='', encoding='utf-8') as data_file:
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(['Timestamp', 'ChangeType', 'Index', 'EEGData'])


    def change_color_sequentially():
        for index in range(table_size):
            for change_type in ['row', 'col']:
                color = 'red' if random.random() < 0.5 else 'white'
                if change_type == 'row':
                    for text_index in range(index * table_size, (index + 1) * table_size):
                        text_objects[text_index].color = color
                else:
                    for text_index in range(index, table_size ** 2, table_size):
                        text_objects[text_index].color = color

                sync_square.fillColor = 'white' if color == 'red' else 'black'
                draw_and_flip()
                time.sleep(1)

                if arduino.inWaiting() > 0:
                    eeg_data = arduino.readline()

                    print("Полученные данные с Arduino:", eeg_data)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    csv_writer.writerow([timestamp, change_type, index, eeg_data])

                reset_colors()
                draw_and_flip()


    def reset_colors():
        for text in text_objects:
            text.color = 'black'
        sync_square.fillColor = 'black'


    def draw_and_flip():
        for text in text_objects:
            text.draw()
        sync_square.draw()
        win.flip()


    psychopy.event.waitKeys(keyList=['space'])

    try:
        while not psychopy.event.getKeys(keyList=['escape']):
            change_color_sequentially()
    finally:
        arduino.close()
        win.close()
