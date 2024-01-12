import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(n):  # С‡С‚РµРЅРёРµ С„Р°Р№Р»РѕРІ
    m = []
    t = []
    with open(n) as nt:
        for line in nt:
            v = line.strip()
            l, cc, k = v.split(';')
            m.append(float(cc))
            t.append(float(l))
    return pd.DataFrame(m), pd.DataFrame(t)


kgr, m = read(input('РІРІРµРґРёС‚Рµ РёРјСЏ С„Р°Р№Р»Р°:'))  # РІРІРµСЃС‚Рё РЅР°Р·РІР°РЅРёРµ С„Р°Р№Р»Р°
# 9_3_2023_10_36_44_portC_dev_EKG (1).csv РїРµСЂРІС‹Р№ Р°РІР°С‚Р°СЂ
# 9_3_2023_10_52_29_portC_dev_EKG.csv   РІС‚РѕСЂРѕР№ Р°РІР°С‚Р°СЂ

kgr = np.array(kgr)
m = np.array(m)
kgr = kgr[10341:12842]  # РѕР±СЂРµР·Р°РµРј 10 СЃРµРєСѓРЅРґ (10 СЃРµРє *250)
m = m[10341:12842]
ans = 0  # РїРµСЂРµРјРµРЅРЅР°СЏ РґР»СЏ РїРѕРґСЃС‡РµС‚Р° С‡РёСЃР»Р° РїРёРєРѕРІ
i = 1
lines = []
while i < len(kgr) - 10:
    chunk = kgr[i:i + 10]
    if max(chunk) - min(chunk) > 0.3 and chunk.argmax() < chunk.argmin():
        lines.append(i + chunk.argmax())
        i += 50
    i += 1
s = 'С‡РёСЃР»Рѕ СЃРµСЂРґРµС‡РЅС‹С… СЃРѕРєСЂР°С‰РµРЅРёР№(РїРёРєРѕРІ):' + str(len(lines))
x, y = [], []
for i in range(len(lines)):
    x.append(kgr[lines[i]])
    y.append(m[lines[i]])
fig = plt.figure()
ax = fig.gca()
ax.plot(m, kgr)
for i in range(len(y)):
    ax.scatter(y[i], x[i], c='#2ca02c')
ax.set_xlabel('РїРµСЂРІС‹Р№ Р°РІР°С‚Р°p    ' + s)
plt.show()  # РІРёР·СѓР°Р»РёР·Р°С†РёСЏ Р·СѓР±С†РѕРІ
