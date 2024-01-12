# -*- coding: utf-8 -*- 

from psychopy import visual, core, event

mywin = visual.Window([600,600],monitor="testMonitor", units="deg")

fixation = visual.GratingStim(win=mywin, size=5, pos=[0,0], sf=0, rgb=-1)

fixation.draw()
mywin.flip()

#чистка
core.wait(2)