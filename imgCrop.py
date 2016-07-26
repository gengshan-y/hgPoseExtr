####
 # Filename:        imgCrop.py
 # Date:            Jul 26 2016
 # Last Edited by:  Gengshan Yang
 # Description:     Crop images wrt given center and scale and target resolution.
 #                  Cropped area = scale*200 X scale*200 pixels
 #                  Time distribution: 
 #                      read image ~30%, crop ~40%, write image ~30%
 ####

import scipy.misc as imLib  # to read and write image
from imgCropTool import *
import string
import os
import sys  # to pass cmd arguments
import time

beg0 = time.time()
accum = [0, 0, 0]

batchSize = int(sys.argv[1])
jobID = int(sys.argv[2]) + 1

with open('/home/gengshan/workJul/darknet/results/comp4_det_test_person.txt', 'r') as f:
    wholeDict = f.readlines()

print('whole list loaded. use ' + str(time.time() - beg0) + 's')
print('imgID\tprob\tfrom\tto')

for it, oriDict in enumerate(wholeDict):
    if it < batchSize * (jobID-1):
        continue
    if it >= batchSize * jobID:
        break
    
    beg1 = time.time()
    ''' Get image parameters '''
    oriDict = oriDict.split('\t')
    img = imLib.imread(oriDict[0])
    wd = float(oriDict[2])
    ht = float(oriDict[3])
    scale = float(oriDict[4])
    accum[0] = accum[0] + time.time() - beg1

    beg1 = time.time()
    ''' Crop image '''
    newImg = crop(img ,[wd, ht], scale, 256)  # width, height
    accum[1] = accum[1] + time.time() - beg1

    ''' Generate new path '''
    newPath = string.replace(oriDict[0], 'tmp', 'poseTmp')
    if not os.path.isdir(string.replace(newPath, oriDict[0].split('/')[-1], '')):
        os.makedirs(string.replace(newPath, oriDict[0].split('/')[-1], ''))
    
    beg1 = time.time()
    ''' Save image '''
    imLib.imsave(newPath, newImg)
    accum[2] = accum[2] + time.time() - beg1

    ''' print information to stderr '''
    eprint(str(it) + '\t' + oriDict[1] + '\t' + oriDict[0] + '\t' + newPath)

print('total time: ' + str(time.time() - beg0))
print('submodel time:')
print(accum)
