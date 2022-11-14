import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import math

def plotForce(file, ite, x, nbParts, name):
    fValues = np.loadtxt(file+str(ite)+'.txt').T
    condition = nbParts!=0
    xValues= fValues[0,condition]
    yValues = fValues[1,condition]
    #print(np.amax(fValues[0]))
    norm = np.linalg.norm(fValues, axis=0)
    print('norm : {}'.format(np.amax(norm)))
    #print('norm2 : {}'.format(fValues[0,ind]*fValues[0,ind]+fValues[1,ind]*fValues[1,ind]))
    #ind = np.argmax(fValues[1])
    #print('norm2 : {}'.format(fValues[0,ind]*fValues[0,ind]+fValues[1,ind]*fValues[1,ind]))
    #print(np.amax(fValues[1]))
    x = x[condition]
    nbParts = nbParts[condition]
    fig, ax = plt.subplots()
    #ax.scatter(x, xValues)
    #fig.canvas.set_window_title(name +' X')
    #ax.set_title(name +' X')
    ax.scatter(xValues, yValues, marker='+')
    fig.canvas.set_window_title(name)
    ax.set_title(name)
    plt.show()
    #fig, ax = plt.subplots()
    #ax.scatter(x, yValues)
    #fig.canvas.set_window_title(name +' Y')
    #ax.set_title(name+' Y')
    #plt.show()

plotPress = False
plotVisc = False
plotExt = False
plotMin = False
plotMax = False
plotAvg = False
parser = argparse.ArgumentParser()
parser.add_argument('--press', action="store_true")
parser.add_argument('--visc', action="store_true")
parser.add_argument('--ext', action="store_true")
parser.add_argument('--min', action="store_true")
parser.add_argument('--max', action="store_true")
parser.add_argument('--avg', action="store_true")
parser.add_argument('--all', action="store_true")
parser.add_argument('--ite')
args = parser.parse_args()
if args.press:
    plotPress = True
if args.visc:
    plotVisc = True
if args.ext:
    plotExt = True
if args.min:
    plotMin = True
if args.max:
    plotMax = True
if args.avg:
    plotAvg = True
if args.all or not (plotMin or plotMax or plotAvg or plotPress or plotVisc or plotExt):
    plotMin = True
    plotMax = True
    plotAvg = True

if args.ite:
    nbIte = int(args.ite)
else:
    nbIte = 1
buildDir = 'build'
logDir = 'log'#os.path.join(buildDir,'log')
forceDir = os.path.join(logDir, 'force')
fPMinFile = os.path.join(forceDir,'fPMinite')
fPMaxFile = os.path.join(forceDir,'fPMaxite')
fPAvgFile = os.path.join(forceDir,'fPAvgite')
fVMinFile = os.path.join(forceDir,'fVMinite')
fVMaxFile = os.path.join(forceDir,'fVMaxite')
fVAvgFile = os.path.join(forceDir,'fVAvgite')
fEMinFile = os.path.join(forceDir,'fEMinite')
fEMaxFile = os.path.join(forceDir,'fEMaxite')
fEAvgFile = os.path.join(forceDir,'fEAvgite')
nbPartFile = os.path.join(logDir,'nbPart.txt')
configFile = os.path.join(logDir,'config.txt')

with open(configFile, 'r') as reader:
    for line in reader:
        print(line, end='')
nbPart = np.loadtxt(nbPartFile)
nbPixel = np.arange(nbPart.shape[0])
print('nbPixel : {}'.format(len(nbPixel)))
for i in range(nbIte,nbIte+1):
    if plotMin or plotPress:
        plotForce(fPMinFile, i, nbPixel, nbPart, 'Pressure min')
    if plotMax or plotPress:
        plotForce(fPMaxFile, i, nbPixel, nbPart, 'Pressure max')
    if plotAvg or plotPress:
        plotForce(fPAvgFile, i, nbPixel, nbPart, 'Pressure avg')
    if plotMin or plotVisc:
        plotForce(fVMinFile, i, nbPixel, nbPart, 'Viscosity min')
    if plotMax or plotVisc:
        plotForce(fVMaxFile, i, nbPixel, nbPart, 'Viscosity max')
    if plotAvg or plotVisc:
        plotForce(fVAvgFile, i, nbPixel, nbPart, 'Viscosity avg')
    if plotMin or plotExt:
        plotForce(fEMinFile, i, nbPixel, nbPart, 'External min')
    if plotMax or plotExt:
        plotForce(fEMaxFile, i, nbPixel, nbPart, 'External max')
    if plotAvg or plotExt:
        plotForce(fEAvgFile, i, nbPixel, nbPart, 'External avg')
