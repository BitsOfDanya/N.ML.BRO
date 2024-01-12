from matplotlib.pyplot import plot

m = open("ДанныеОльги_portA_dev_EEG.csv", "r").read().split("\n")[:-1]
time = [] #временные отметки
sig = [] #сигнал
com = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m)):
    st = m[i].split(";")
    time.append(float(st[0]))
    sig.append(float(st[1]))
    com.append(int(st[2]))
plot(sig)
show()

m2 = open("Данные Михаила_portC_dev_EKG.csv", "r").read().split("\n")[:-1]
time1 = [] #временные отметки
sig1 = [] #сигнал
com1 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m2)):
    st1 = m2[i].split(";")
    time1.append(float(st1[0]))
    sig1.append(float(st1[1]))
    com1.append(int(st1[2]))
plot(sig1)
show()

m3 = open("Данные Михаила_portN1_dev_KGR.csv", "r").read().split("\n")[:-1]
time2 = [] #временные отметки
sig2 = [] #сигнал
com2 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m3)):
    st2 = m3[i].split(";")
    time2.append(float(st2[0]))
    sig2.append(float(st2[1]))
    com2.append(int(st2[2]))
plot(sig2)
show()

m4 = open("ДанныеОльги_portN2_dev_BRL.csv", "r").read().split("\n")[:-1]
time3 = [] #временные отметки
sig3 = [] #сигнал
com3 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m4)):
    st3 = m4[i].split(";")
    time3.append(float(st3[0]))
    sig3.append(float(st3[1]))
    com3.append(int(st3[2]))
plot(sig3)
show()

m5 = open("Данные Михаила_portA_dev_EEG.csv", "r").read().split("\n")[:-1]
time4 = [] #временные отметки
sig4 = [] #сигнал
com4 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m5)):
    st4 = m5[i].split(";")
    time4.append(float(st4[0]))
    sig4.append(float(st4[1]))
    com4.append(int(st4[2]))
plot(sig4)
show()

m6 = open("Данные Михаила_portC_dev_EKG.csv", "r").read().split("\n")[:-1]
time5 = [] #временные отметки
sig5 = [] #сигнал
com5 = [] #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m6)):
    st5 = m6[i].split(";")
    time5.append(float(st5[0]))
    sig5.append(float(st5[1]))
    com5.append(int(st5[2]))
plot(sig5)
show()

m7 = open("Данные Михаила_portN1_dev_KGR.csv", "r").read().split("\n")[:-1]
time6 = [] #временные отметки
sig6 = []  #сигнал
com6 = []  #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m7)):
    st6 = m7[i].split(";")
    time6.append(float(st6[0]))
    sig6.append(float(st6[1]))
    com6.append(int(st6[2]))
plot(sig6)
show()

m8 = open("Данные Михаила_portN2_dev_BRL.csv", "r").read().split("\n")[:-1]
time7 = [] #временные отметки
sig7 = []  #сигнал
com7 = []  #состояние(ответ на стимул, отпраленный через питон)
for i in range(len(m8)):
    st7 = m8[i].split(";")
    time7.append(float(st7[0]))
    sig7.append(float(st7[1]))
    com7.append(int(st7[2]))
plot(sig7)
show()
