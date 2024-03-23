from psychopy import visual, core, event
import random

win = visual.Window([800, 600], color='white')
message = "Я-скрипт,\nи я вывожу\n на экран буквы"
text_positions = [(i % 20 * 0.07 - 0.7, 0.2 - 0.1 * (i // 20)) for i in range(len(message))]

def random_color():
    return [random.uniform(-1, 1) for _ in range(3)]

color_change_phase = False
position_change_phase = False

while not event.getKeys(keyList=['escape']):
    events = event.getKeys()
    if 'space' in events and color_change_phase:
        position_change_phase = True
    elif events:
        color_change_phase = True
    
    square = visual.Rect(win, width=0.1, height=0.1, pos=(0.7, -0.4), fillColor='black', lineColor='black')
    
    if color_change_phase and not position_change_phase:
        chosen_index = random.randint(0, len(message.replace('\n', '')) - 1)
        for i, char in enumerate(message.replace('\n', '')):
            color = random_color() if i == chosen_index else 'black'
            visual.TextStim(win, text=char, pos=text_positions[i], color=color, height=0.05).draw()
        square.fillColor = 'white'
    elif position_change_phase:
        chosen_index = random.randint(0, len(message.replace('\n', '')) - 1)
        for i, char in enumerate(message.replace('\n', '')):
            if i == chosen_index:
                pos = (text_positions[i][0] + random.uniform(-0.1, 0.1), text_positions[i][1] + random.uniform(-0.1, 0.1))
                color = random_color()
            else:
                pos = text_positions[i]
                color = 'black'
            visual.TextStim(win, text=char, pos=pos, color=color, height=0.05).draw()
        square.fillColor = 'white'
    else:
        for i, char in enumerate(message.replace('\n', '')):
            visual.TextStim(win, text=char, pos=text_positions[i], color='black', height=0.05).draw()
        square.fillColor = 'black'
    
    square.draw()
    win.flip()
    core.wait(0.5)

win.close()
