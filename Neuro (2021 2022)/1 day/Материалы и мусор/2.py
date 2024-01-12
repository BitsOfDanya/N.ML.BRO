from psychopy import visual, core, event

mywin = visual.Window([600,600],monitor="testMonitor", units="deg")
stim = visual.TextStim(mywin, 'сепулька',color=(0, 1, 0), colorSpace='rgb')

while True:
    stim.draw()
    mywin.flip()

    if len(event.getKeys())>0:
        break
    event.clearEvents()