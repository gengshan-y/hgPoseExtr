# hgPoseExtr
Modified hourglass. Extract pose estimations from persion detection results.

* To crop images using detection results, run

    `python imgCrop.py batch_size 0-based_taskid` e.g.: `python imgCrop.py 10000 0`
or just run

    `condor_submit condor.submit`

* To run hourglss model on GPU, run

    `th mulThreadsExtr.lua [options]`  

    e.g.:  `th mulThreadsExtr.lua -iter 200 -outname hg_img -GPU_num 2`

To check the options, run

`th mulThreadsExtr.lua -help`

## About notebooks
* imgCrop.ipynb is the experimental version of imgCrop.py, but used the same method.
* poseFeatExtr-yolo.ipynb is the experimental version for pose estimation (single-threaded).
* img.lua, util.lua and poseExtrUtil.lua are only used by poseFeatExtr-yolo.ipynb

## Notes
Cpu machines crop the images, and GPU machines run neural networks. Currently need to calculate the number of batches manually before running.

### Garbage collection
Garbage-free. Constant usage of 1% memory.

### Speed
~42fps on Tesla 40[2 devs], batch_size=10
~80fps on Titan x[4 devs], batch_size=5

### Merge h5 files
Use mergeH5.ipynb to merge .h5 files created by different threads.

### Visualize generated pose features
Use featVisualize.ipynb to generate pose-embedded videos.
