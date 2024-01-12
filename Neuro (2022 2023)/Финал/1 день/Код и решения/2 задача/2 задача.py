# -*- coding: utf-8 -*- 

from psychopy import visual, core, event

mywin = visual.Window([600,800],monitor="testMonitor", units="deg")
grating = visual.GratingStim(win=mywin, mask='circle', size=7, pos=[0,0] )
grating.draw()
mywin.flip()
core.wait(2)
    
mywin = visual.Window([600,800],monitor="testMonitor", units="deg")
image = visual.ImageStim(win=mywin,name='image',image='КОТ.jpg', mask=None, ori=45.0, pos=(0, 0), size=(8, 8),color=[1,1,1], colorSpace='rgb', opacity=None, flipHoriz=False, flipVert=False,texRes=128.0, interpolate=True)

while True:
    image.draw()
    mywin.flip()
    if event.getKeys(keyList = ['d']):
        break

mywin.close()
core.quit()
