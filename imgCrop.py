####
 # Filename:        imgCrop.py
 # Date:            Jul 26 2016
 # Last Edited by:  Gengshan Yang
 # Description:     Crop images wrt given center and scale and target resolution.
 #                  Cropped area = scale*200 X scale*200 pixels
 #                  10k images takes 20min on blade16
 #                  Time distribution:
 #                      read image ~38%, crop ~24%, write image ~38%
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
    idx = 0
    wholeDict = []
    for line in f:
        
        if idx < batchSize * (jobID-1):
            idx += 1
            continue
        if idx >= batchSize * jobID:
            break
        wholeDict.append(str(idx) + "\t" + line)
        idx += 1

print('whole list loaded. use ' + str(time.time() - beg0) + 's')
print('imgID\tprob\tfrom\tto')

for oriDict in wholeDict:
    beg1 = time.time()
    ''' Get image parameters '''
    oriDict = oriDict.split('\t')
    img = imLib.imread(oriDict[1])
    wd = float(oriDict[3])
    ht = float(oriDict[4])
    scale = float(oriDict[5])
    accum[0] = accum[0] + time.time() - beg1

    beg1 = time.time()
    ''' Crop image '''
    newImg = crop(img ,[wd, ht], scale, 256)  # width, height
    accum[1] = accum[1] + time.time() - beg1

    ''' Generate new path '''
    newPath = string.replace(oriDict[1], 'tmp', 'poseTmp')
    if not os.path.isdir(string.replace(newPath, oriDict[1].split('/')[-1], '')):
        os.makedirs(string.replace(newPath, oriDict[1].split('/')[-1], ''))
    
    beg1 = time.time()
    ''' Save image '''
    imLib.imsave(newPath, newImg)
    accum[2] = accum[2] + time.time() - beg1

    ''' print information to stderr '''
    eprint(oriDict[0] + '\t' + oriDict[2] + '\t' + oriDict[1] + '\t' + newPath)

print('total time: ' + str(time.time() - beg0))
print('submodel time:')
print(accum)
