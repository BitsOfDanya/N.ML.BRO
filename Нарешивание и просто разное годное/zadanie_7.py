import numpy as np
import random
from sklearn import discriminant_analysis as LDA
from sklearn import preprocessing
from scipy import signal

# ������� ��� ������������� ������, 
# �������� ������ ������ ������������� ����� ����������� �������� ���� �������� ���������� �������� � �������������
def epoProc(DATA, epoStart):
    Ep = DATA[epoStart+50:epoStart+300, :6]
    ep = signal.decimate(Ep, 5, ftype='fir', axis=0)
    ep -= ep[0]
    ep = ep.reshape(ep.shape[0]*ep.shape[1])

    return ep

# ��������� ��������� � �������� �������
Train_src = np.loadtxt('P300_Train.txt')
Test_src = np.loadtxt('P300_Test.txt')

# ������ ����� ���������� ������, � ������: 301 ~= 250* (400+800)/1000
epo_len = 301

# ��������� �������, � ������� ���������� ������ ����� ��������� - "�����"
# ������� �������, � ������� Label �������� � 0 �� 1, �.�. ��������� ������ P300
Start_P300 = np.nonzero((Train_src[1:, -1] - Train_src[:-1, -1]) == 1)[0]
# ������� �������, � ������� Label �������� � 1 �� 0, �.�. ������� P300 �� ���������
Start_nonP300 = np.nonzero((Train_src[1:, -1] - Train_src[:-1, -1]) == -1)[0]

# �������� ����-�� �300 � ������ ����� ��� ��� � ��������� ��� � ��������������� ������
if Train_src[0, -1]:
    Start_P300 = np.append(Start_P300, 0)
else:
    Start_nonP300 = np.append(Start_nonP300, 0)

# ������� ������� ���������������� ������ ��� �������� �������������� 
Target = np.array([epoProc(Train_src, i) for i in Start_P300])
NonTarget = np.array([epoProc(Train_src, i) for i in Start_nonP300])
X = preprocessing.scale(np.vstack((Target, NonTarget)))
y = np.hstack((np.ones(Target.shape[0]), np.zeros(NonTarget.shape[0])))

# ������� ������������� � ������� ���
cls = LDA.LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage='auto')
cls.fit(X, y)

# ��������� �������� ������ � ������� �� ��� �������������
TestEpos = list(range(0, Test_src.shape[0], epo_len))
Tests = np.array([epoProc(Test_src, i) for i in TestEpos])

# ���� � �������� ������ �300 � ������� ���������
res = cls.predict_proba(Tests)
Answ = res.argmax(axis=1)
print(Answ)

