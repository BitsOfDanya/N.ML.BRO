
from psychopy import visual, core, event
import random

win = visual.Window([800, 600], color="white")

text = "Я - скрипт, и я вывожу на экран буквы"
text_objects = [visual.TextStim(win, text=char, pos=(i*20 - len(text)*10/2, 0), color="black") for i, char in enumerate(text)]

square = visual.Rect(win, width=50, height=50, pos=[300, -200], fillColor="black", lineColor="black")

for obj in text_objects:
    obj.draw()
square.draw()
win.flip()

def change_color_and_position(change_position=False):
    indices = list(range(len(text_objects)))
    random.shuffle(indices)
    for index in indices:
        original_pos = text_objects[index].pos
        text_objects[index].color = random.choice(["red", "green", "blue", "yellow"])
        if change_position:
            text_objects[index].pos = (random.randint(-350, 350), random.randint(-250, 250))
        square.fillColor = "white"
        draw_all()
        core.wait(0.5)
        text_objects[index].color = "black"
        text_objects[index].pos = original_pos
        square.fillColor = "black"
        draw_all()
        core.wait(0.5)

def draw_all():
    for obj in text_objects:
        obj.draw()
    square.draw()
    win.flip()

space_pressed = False
while True:
    keys = event.getKeys()
    for key in keys:
        if key == 'escape':
            win.close()
            core.quit()
        elif key == 'space':
            space_pressed = True
        else:
            change_color_and_position(change_position=space_pressed)
