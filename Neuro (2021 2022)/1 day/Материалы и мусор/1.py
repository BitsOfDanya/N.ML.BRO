from psychopy import *
mywin = visual.Window([600, 600], monitor='testMonitor', units='deg')

while True:
    mywin.flip()
    
mywin.close()
core.quit()
