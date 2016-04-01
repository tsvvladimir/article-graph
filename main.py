# -*- coding: utf-8 -*-
from __future__ import division

abaselinex = []
abaseliney = []
abaseline_activex = []
abaseline_activey = []
amarginx = []
amarginy = []
aclusterx = []
aclustery = []
aclusterz = []
with open('fold1.txt') as f:
    content = f.readlines()
    for idx, line in enumerate(content):
        if line.startswith('baseline_solution'):
            cnt = line.split()
            abaselinex.append(float(cnt[1]))
            abaseliney.append(float(cnt[3]))
        if line.startswith('baseline_active'):
            cnt = line.split()
            abaseline_activex.append((float(cnt[1])))
            abaseline_activey.append(float(cnt[3]))
        if line.startswith('active_minimum'):
            cnt = line.split()
            amarginx.append(float(cnt[1]))
            amarginy.append(float(cnt[3]))
        if line.startswith('active_cluster'):
            cnt = line.split()
            aclusterx.append(float(cnt[1]))
            aclustery.append(float(cnt[3]))
            aclusterz.append(float(cnt[5]))

bbaselinex = []
bbaseliney = []
bbaseline_activex = []
bbaseline_activey = []
bmarginx = []
bmarginy = []
bclusterx = []
bclustery = []
bclusterz = []
with open('fold2.txt') as f:
    content = f.readlines()
    for idx, line in enumerate(content):
        if line.startswith('baseline_solution'):
            cnt = line.split()
            bbaselinex.append(float(cnt[1]))
            bbaseliney.append(float(cnt[3]))
        if line.startswith('baseline_active'):
            cnt = line.split()
            bbaseline_activex.append((float(cnt[1])))
            bbaseline_activey.append(float(cnt[3]))
        if line.startswith('active_minimum'):
            cnt = line.split()
            bmarginx.append(float(cnt[1]))
            bmarginy.append(float(cnt[3]))
        if line.startswith('active_cluster'):
            cnt = line.split()
            #print idx
            #print cnt
            bclusterx.append(float(cnt[1]))
            bclustery.append(float(cnt[3]))
            bclusterz.append(float(cnt[5]))


cbaselinex = []
cbaseliney = []
cbaseline_activex = []
cbaseline_activey = []
cmarginx = []
cmarginy = []
cclusterx = []
cclustery = []
cclusterz = []
with open('fold3.txt') as f:
    content = f.readlines()
    for idx, line in enumerate(content):
        if line.startswith('baseline_solution'):
            cnt = line.split()
            cbaselinex.append(float(cnt[1]))
            cbaseliney.append(float(cnt[3]))
        if line.startswith('baseline_active'):
            cnt = line.split()
            cbaseline_activex.append((float(cnt[1])))
            cbaseline_activey.append(float(cnt[3]))
        if line.startswith('active_minimum'):
            cnt = line.split()
            cmarginx.append(float(cnt[1]))
            cmarginy.append(float(cnt[3]))
        if line.startswith('active_cluster'):
            cnt = line.split()
            cclusterx.append(float(cnt[1]))
            cclustery.append(float(cnt[3]))
            cclusterz.append(float(cnt[5]))

dbaselinex = []
dbaseliney = []
dbaseline_activex = []
dbaseline_activey = []
dmarginx = []
dmarginy = []
dclusterx = []
dclustery = []
dclusterz = []
with open('fold4.txt') as f:
    content = f.readlines()
    for idx, line in enumerate(content):
        if line.startswith('baseline_solution'):
            cnt = line.split()
            dbaselinex.append(float(cnt[1]))
            dbaseliney.append(float(cnt[3]))
        if line.startswith('baseline_active'):
            cnt = line.split()
            dbaseline_activex.append((float(cnt[1])))
            dbaseline_activey.append(float(cnt[3]))
        if line.startswith('active_minimum'):
            cnt = line.split()
            dmarginx.append(float(cnt[1]))
            dmarginy.append(float(cnt[3]))
        if line.startswith('active_cluster'):
            cnt = line.split()
            dclusterx.append(float(cnt[1]))
            dclustery.append(float(cnt[3]))
            dclusterz.append(float(cnt[5]))


import numpy as np

baselinex = np.add(abaselinex, np.add(bbaselinex, np.add(cbaselinex, dbaselinex)))
baseliney = np.add(abaseliney, np.add(bbaseliney, np.add(cbaseliney, dbaseliney)))
baseline_activex = np.add(abaseline_activex, np.add(bbaseline_activex, np.add(cbaseline_activex, dbaseline_activex)))
baseline_activey = np.add(abaseline_activey, np.add(bbaseline_activey, np.add(cbaseline_activey, dbaseline_activey)))
#print len(amarginx)
#print len(bmarginx)
#print len(cmarginx)
#print amarginx
#print bmarginx
marginx = np.add(amarginx, np.add(bmarginx, np.add(cmarginx, cmarginy)))
marginy = np.add(amarginy, np.add(bmarginy, np.add(cmarginy, dmarginy)))
clusterx = np.add(aclusterx, np.add(bclusterx, np.add(cclusterx, dclusterx)))
clustery = np.add(aclustery, np.add(bclustery, np.add(cclustery, dclustery)))
clusterz = np.add(aclusterz, np.add(bclusterz, np.add(cclusterz, dclusterz)))



#print baselinex

baselinex = np.divide(baselinex, 4)
baseliney = np.divide(baseliney, 4)
baseline_activex = np.divide(baseline_activex, 4)
baseline_activey = np.divide(baseline_activey, 4)
marginx = np.divide(marginx, 4)
marginy = np.divide(marginy, 4)
clusterx = np.divide(clusterx, 4)
clustery = np.divide(clustery, 4)
clusterz = np.divide(clusterz, 4)

import numpy as np
import matplotlib.pyplot as plt

#print abaselinex, bbaselinex, cbaselinex
basex = [0, baselinex[0]]
basey = [baseliney[0], baseliney[0]]

#plt.plot(basex, basey, 'r', clusterx, clustery, 'b')
#plt.show()

from matplotlib import rc
font = {'family': 'Droid Sans',
        'weight': 'normal',
        'size': 14}

'''
noactive, = plt.plot(basex, basey, label=u'without active learning', linestyle='--')
active, = plt.plot(clusterx, clustery, label = u'with active learning', linestyle='-')

legend = plt.legend(handles=[noactive, active], loc=4)

ax = plt.gca().add_artist(legend)
plt.xlabel(u'train set size')
plt.ylabel(u'f1 score')

plt.show()
'''

print 'baseline f1 score:', basey[1]
print 'maximum f1 score:', clustery.max(), 'at train size', clusterx[np.argmax(clustery)], 'and proba', clusterz[np.argmax(clustery)]

crit = []
for i, proba in enumerate(clusterz):
    if proba > clusterz[np.argmax(clustery)]:
        crit.append(i)

crit_f1 = []
for i in range(0, len(crit)):
    crit_f1.append(clustery[crit[i]])

print 'f1 score when crit satisfied:', crit_f1


#clusterzder = np.diff(clusterz)
#plt.plot(clusterx[:len(clusterx) - 1], clusterzder, 'r', clusterx[:len(clusterx) - 1], clustery[:len(clustery) - 1], 'g')
#plt.plot(clusterx[:len(clusterx) - 1], clusterzder)
#plt.show()