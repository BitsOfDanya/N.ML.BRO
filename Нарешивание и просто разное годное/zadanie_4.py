#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# ��������� ������ �� �����
D = np.loadtxt('P300_dataSet.txt')

# ������� �������, � ������� Label �������� � 0 �� 1, �.�. ��������� ������ P300
T_starts = np.nonzero((D[1:, -1] - D[:-1, -1]) == 1)[0]
# ������������, ������� ����� �������� �� ���������� ����� ����� 400�� + 800�� ��� ������� ��������� 250��
epo_len = int( 250 * ( (800+400)/1000 ) )

T = np.zeros((epo_len, 6))
# ��������� ��� ������� ������ ("�����"), � ������� ��������� �300
for t in T_starts:
    T += D[t:t+epo_len, :-4]

T = T / len(T_starts)

# ������ ���������� ��������� ������
#handle = plt.plot(lab[1]['index'], lab[1].drop('index', axis=1))
handle = plt.plot(T)
plt.legend(handles=handle, labels=['Cz', 'Pz', 'PO7', 'PO8', 'O1', 'O2'], loc=1)
plt.ylabel('mV')
plt.xlabel('ms')
plt.show()

