#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on February 22, 2022, at 18:13
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'Эксперимент в PsychoPy  № 1'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='A:\\IT_work\\Python\\NTO\\Neuro\\Эксперимент в PsychoPy  № 1_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=(1024, 768), fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "trial"
trialClock = core.Clock()
dots = visual.DotStim(
    win=win, name='dots',
    nDots=100, dotSize=2.0,
    speed=0.1, dir=0.0, coherence=1.0,
    fieldPos=(0.0, 0.0), fieldSize=1.0,fieldShape='circle',
    signalDots='same', noiseDots='direction',dotLife=3.0,
    color=[1.0,1.0,1.0], colorSpace='rgb', opacity=None,
    depth=0.0)
noise = visual.NoiseStim(
    win=win, name='noise',
    noiseImage=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=1.0,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.0625], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise.buildNoise()
text = visual.TextStim(win=win, name='text',
    text='Hello man!',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "trial"-------
continueRoutine = True
routineTimer.add(20.000000)
# update component parameters for each repeat
dots.refreshDots()
# keep track of which components have finished
trialComponents = [dots, noise, text]
for thisComponent in trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "trial"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = trialClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=trialClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *dots* updates
    if dots.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        dots.frameNStart = frameN  # exact frame index
        dots.tStart = t  # local t and not account for scr refresh
        dots.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(dots, 'tStartRefresh')  # time at next scr refresh
        dots.setAutoDraw(True)
    if dots.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > dots.tStartRefresh + 5.0-frameTolerance:
            # keep track of stop time/frame for later
            dots.tStop = t  # not accounting for scr refresh
            dots.frameNStop = frameN  # exact frame index
            win.timeOnFlip(dots, 'tStopRefresh')  # time at next scr refresh
            dots.setAutoDraw(False)
    
    # *noise* updates
    if noise.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
        # keep track of start time/frame for later
        noise.frameNStart = frameN  # exact frame index
        noise.tStart = t  # local t and not account for scr refresh
        noise.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(noise, 'tStartRefresh')  # time at next scr refresh
        noise.setAutoDraw(True)
    if noise.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > noise.tStartRefresh + 8.0-frameTolerance:
            # keep track of stop time/frame for later
            noise.tStop = t  # not accounting for scr refresh
            noise.frameNStop = frameN  # exact frame index
            win.timeOnFlip(noise, 'tStopRefresh')  # time at next scr refresh
            noise.setAutoDraw(False)
    if noise.status == STARTED:
        if noise._needBuild:
            noise.buildNoise()
    
    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 8.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        text.setAutoDraw(True)
    if text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text.tStartRefresh + 12.0-frameTolerance:
            # keep track of stop time/frame for later
            text.tStop = t  # not accounting for scr refresh
            text.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
            text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "trial"-------
for thisComponent in trialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('dots.started', dots.tStartRefresh)
thisExp.addData('dots.stopped', dots.tStopRefresh)
thisExp.addData('noise.started', noise.tStartRefresh)
thisExp.addData('noise.stopped', noise.tStopRefresh)
thisExp.addData('text.started', text.tStartRefresh)
thisExp.addData('text.stopped', text.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
