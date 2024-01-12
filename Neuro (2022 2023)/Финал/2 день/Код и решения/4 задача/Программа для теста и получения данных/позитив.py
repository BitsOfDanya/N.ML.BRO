# -*- coding: utf-8 -*-

from random import shuffle
from psychopy import visual, event
import time
import os
import socket
import ctypes.wintypes

CSIDL_PERSONAL = 5  # My Documents
SHGFP_TYPE_CURRENT = 0  # Get current, not default value

buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
f = str(buf.value) + "\_bitronics_port.txt"
with open(f, "r") as dr:
    a = int(dr.read())
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOTSPORT = a

s.connect(("127.0.0.1", HOTSPORT))


def metka(o):
    s.send((str(o) + "\n").encode("utf-8"))
    time.sleep(1)


files = os.listdir()
imagespos = list(filter(lambda x: x.endswith('positiv.jpg'), files))
mywin = visual.Window([800, 600], monitor="testMonitor", units="pix")  # создаёт окно


def drawim(image, posi=[0, 0]):  # image - адрес изображения, posi - позиция list в формате [x;y]
    if posi is None:
        posi = [0, 0]
    im = visual.ImageStim(mywin, image, ori=0, size=600, pos=posi)  # mask="circle",size=8, pos=posi)
    im.draw()
    mywin.update()


def waiti(time1, time2):  # ждать и обновить окно
    time.sleep(time1)
    mywin.update()
    time.sleep(time2)

images = imagespos + imagesneg
shuffle(images)
for i in range(len(images)):
    x = 0
    y = 0
    metka(1 if images[i].endswith('positiv.jpg') else 0)
    drawim(images[i], [x, y])
    waiti(4, 0)