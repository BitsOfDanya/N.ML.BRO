# -*- coding: utf-8 -*- 

from psychopy import visual, core, event

mywin = visual.Window([600,600],monitor="testMonitor", units="deg")

fixation = visual.GratingStim(win=mywin, size=5, pos=[0,0], sf=0, rgb=-1)

while True:
    fixation.draw()
    mywin.flip()

    if len(event.getKeys())>0:
        break
    event.clearEvents()

#чистка
mywin.close()
core.quit()
