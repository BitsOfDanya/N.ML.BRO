"""
m1 = open("annan_portC_dev_EKG.csv", "r").read().split("\n")[:-1]
time1 = [] #временные отметки
sig1 = [] #сигнал
com1 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m1)):
    st1 = m1[i].split(";")
    time1.append(float(st1[0]))
    sig1.append(float(st1[1]))
    com1.append(int(st1[2]))
plt.xlim([25, 50])
plt.plot(sig1)
plt.show()

m2 = open("annap_portC_dev_EKG.csv", "r").read().split("\n")[:-1]
time2 = [] #временные отметки
sig2 = [] #сигнал
com2 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m2)):
    st2 = m2[i].split(";")
    time2.append(float(st2[0]))
    sig2.append(float(st2[1]))
    com2.append(int(st2[2]))
plt.plot(sig2)
plt.show()

m3 = open("katyan_portC_dev_EKG.csv", "r").read().split("\n")[:-1]
time3 = [] #временные отметки
sig3 = []  #сигнал
com3 = []  #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m3)):
    st3 = m3[i].split(";")
    time3.append(float(st3[0]))
    sig3.append(float(st3[1]))
    com3.append(int(st3[2]))
plt.plot(sig3)
plt.show()

m4 = open("katyap_portC_dev_EKG.csv", "r").read().split("\n")[:-1]
time4 = [] #временные отметки
sig4 = [] #сигнал
com4 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m4)):
    st4 = m4[i].split(";")
    time4.append(float(st4[0]))
    sig4.append(float(st4[1]))
    com4.append(int(st4[2]))
plt.plot(sig4)
plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read(n):
    a = []
    b = []
    with open(n) as nt:
        for line in nt:
            v = line.strip()
            l, cc, k = v.split(';')
            a.append(float(cc))
            b.append(float(l))
    return pd.DataFrame(a), pd.DataFrame(b)

signal, z = read('annan_portC_dev_EKG.csv')

signal = np.array(signal)
z = np.array(z)
signal = signal[10000:12000]
z = z[10000:12000]
ans = 0
i = 1
lines = []
while i < len(signal) - 10:
    chunk = signal[i:i + 10]
    if max(chunk) - min(chunk) > 0.3 and chunk.argmax() < chunk.argmin():
        lines.append(i + chunk.argmax())
        i += 50
    i += 1

s = str(len(lines))
x, y = [], []

for i in range(len(lines)):
    x.append(signal[lines[i]])
    y.append(z[lines[i]])
fig = plt.figure()
ax = fig.gca()
ax.plot(z, signal)

for i in range(len(y)):
    ax.scatter(y[i], x[i], c='000000')

plt.show()

signal, z = read('annap_portC_dev_EKG.csv')

signal = np.array(signal)
z = np.array(z)
signal = signal[10000:12000]
z = z[10000:12000]
ans = 0
i = 1
lines = []
while i < len(signal) - 10:
    chunk = signal[i:i + 10]
    if max(chunk) - min(chunk) > 0.3 and chunk.argmax() < chunk.argmin():
        lines.append(i + chunk.argmax())
        i += 50
    i += 1

s = str(len(lines))
x, y = [], []


for i in range(len(lines)):
    x.append(signal[lines[i]])
    y.append(z[lines[i]])
fig = plt.figure()
ax = fig.gca()
ax.plot(z, signal)

for i in range(len(y)):
    ax.scatter(y[i], x[i], c='000000')

plt.show()

signal, z = read('katyan_portC_dev_EKG.csv')

signal = np.array(signal)
z = np.array(z)
signal = signal[10000:12000]
z = z[10000:12000]
ans = 0
i = 1
lines = []
while i < len(signal) - 10:
    chunk = signal[i:i + 10]
    if max(chunk) - min(chunk) > 0.3 and chunk.argmax() < chunk.argmin():
        lines.append(i + chunk.argmax())
        i += 50
    i += 1

s = str(len(lines))
x, y = [], []


for i in range(len(lines)):
    x.append(signal[lines[i]])
    y.append(z[lines[i]])
fig = plt.figure()
ax = fig.gca()
ax.plot(z, signal)


for i in range(len(y)):
    ax.scatter(y[i], x[i], c='000000')

plt.show()

signal, z = read('katyap_portC_dev_EKG.csv')

signal = np.array(signal)
z = np.array(z)
signal = signal[10000:12000]
z = z[10000:12000]
ans = 0
i = 1
lines = []
while i < len(signal) - 10:
    chunk = signal[i:i + 10]
    if max(chunk) - min(chunk) > 0.3 and chunk.argmax() < chunk.argmin():
        lines.append(i + chunk.argmax())
        i += 50
    i += 1

s = str(len(lines))
x, y = [], []

for i in range(len(lines)):
    x.append(signal[lines[i]])
    y.append(z[lines[i]])
fig = plt.figure()
ax = fig.gca()
ax.plot(z, signal)

for i in range(len(y)):
    ax.scatter(y[i], x[i], c='000000')

plt.show()
