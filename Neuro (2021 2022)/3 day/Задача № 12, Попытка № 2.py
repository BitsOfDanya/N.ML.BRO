# -*- coding: utf-8 -*- 

from psychopy import visual,  event
from socket import socket, AF_INET, SOCK_STREAM
from time import sleep

#настройки сервера для связи с битрониксом
marks = []
host = "127.0.0.1"
port = int(open("C:/Users/"+user+"/Documents/_bitronics_port.txt", "r").read())
print(port)
s = socket(AF_INET, SOCK_STREAM)
s.connect((host, port))

#окошко и передача данных в битроникс
mywin = visual.Window([600,600],monitor="testMonitor", units="deg")
for i in range(1, 6):
    d = str(i)+"\n"
    s.send(d.encode())
    image = visual.ImageStim(win=mywin,name='image',image=str(i)+'.png', mask=None, ori=0.0, pos=(0, 0), size=(8, 8),color=[1,1,1], colorSpace='rgb', opacity=None,flipHoriz=False, flipVert=False,texRes=128.0, interpolate=True)
    image.draw()
    mywin.flip()
    sleep(2)
    event.clearEvents()
mywin.close()
s.send("0\n".encode())
s.close()
