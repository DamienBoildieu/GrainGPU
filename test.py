import subprocess as sp
import numpy as np
import os
import matplotlib.pyplot as plt

sizePowMin = 1
sizePowMax = 7#11
iteMax = 16
sizeRange = [2**i for i in range(sizePowMin,sizePowMax)]
iteRange = [i*5 for i in range(1,iteMax)]
nbTests = len(sizeRange)*len(iteRange)
print('================================================================================')
print('{} tests will be done, width and height values are from 2^{} to 2^{} \
and number of iterations are from 5 to {}'.format(nbTests, sizePowMin, sizePowMax, (iteMax-1)*5))
print('================================================================================')
execFile = 'Grains'
buildDir = 'build'
nbTest = 0
for size in sizeRange:
    for nbIte in iteRange:
        print('Test {} -- nbPixels : {} -- nbIte : {}'.format(nbTest, size*size, nbIte))
        opt=' -r {} {} -ite {}'.format(size, size, nbIte)
        opt = ' -r {} {} -ite {} -ut'.format(size, size, nbIte)
        if (nbIte==iteRange[0]):
            opt += ' -ct -it'
        cmd = os.path.join('.',buildDir,execFile+opt)
        returned_value = sp.call(cmd, shell=True)
        nbTest += 1

chronoDir = 'chrono'
initFile = 'init.txt'
updateFile = 'update.txt'
convoluateFile = 'convo.txt'
print('================================================================================')
print('ReadFiles')
initTimes = np.loadtxt(os.path.join(chronoDir,initFile)).T
minVal = np.min(initTimes[3])
maxVal = np.max(initTimes[3])
mean = np.mean(initTimes[3])
median = np.median(initTimes[3])
std = np.std(initTimes[3])
print('Init min : {} -- max : {} -- mean : {} -- median : {} -- std : {}'.format(minVal, maxVal, mean, median, std))

updateTimes = np.loadtxt(os.path.join(chronoDir,updateFile)).T
minVal = np.min(updateTimes[4])
maxVal = np.max(updateTimes[4])
mean = np.mean(updateTimes[4])
median = np.median(updateTimes[4])
std = np.std(updateTimes[4])
print('Update min : {} -- max : {} -- mean : {} -- median : {} -- std : {}'.format(minVal, maxVal, mean, median, std))

convoluateTimes = np.loadtxt(os.path.join(chronoDir,convoluateFile)).T
minVal = np.min(convoluateTimes[3])
maxVal = np.max(convoluateTimes[3])
mean = np.mean(convoluateTimes[3])
median = np.median(convoluateTimes[3])
std = np.std(convoluateTimes[3])
print('Convo min : {} -- max : {} -- mean : {} -- median : {} -- std : {}'.format(minVal, maxVal, mean, median, std))

figSize = (25,20)

initAbs = np.vstack([initTimes[0], initTimes[1], initTimes[2]]).T
updateAbs = np.vstack([updateTimes[0], updateTimes[1], updateTimes[2]]).T
convAbs = np.vstack([convoluateTimes[0], convoluateTimes[1], initTimes[2]]).T

colormap = plt.cm.get_cmap('jet')
colors = [colormap(i) for i in np.linspace(0, 1, initTimes.shape[1])]
print('Plot init durations')
initPos = np.arange(initTimes.shape[1])
plt.figure(figsize=figSize)
plt.bar(initPos, initTimes[3], color=colors)
plt.ylabel('Durations (s)')
plt.xticks(initPos, initAbs, rotation='vertical')
plt.xlabel('Image Dimensions')
plt.savefig(os.path.join(chronoDir,'init.png'), bbox_inches='tight')
plt.close()

print('Plot update durations')
unique = np.unique(updateAbs, axis=0)
nbGroup = len(unique)
nbIte = len(iteRange)
divided = np.array(np.split(updateTimes[4], nbGroup)).T

width = 0.8
widthIte = width/nbIte
x = np.arange(nbGroup)
plt.figure(figsize=figSize)
for i in range(nbIte):
    plt.bar(x-width*0.5+i*widthIte+widthIte*0.5, divided[i], widthIte)
plt.ylabel('Durations (s)')
plt.xlabel('Image Dimensions and iteration number')
plt.xticks(x, unique, rotation='vertical')
plt.savefig(os.path.join(chronoDir,'update.png'), bbox_inches='tight')
plt.close()

print('Plot convoluate durations')
convPos = np.arange(convoluateTimes.shape[1])
plt.figure(figsize=figSize)
plt.bar(convPos, convoluateTimes[3], color=colors)
plt.ylabel('Durations (s)')
plt.xticks(convPos, convAbs, rotation='vertical')
plt.xlabel('Image Dimensions')
plt.savefig(os.path.join(chronoDir,'conv.png'), bbox_inches='tight')
plt.close()
print('End')
print('================================================================================')
