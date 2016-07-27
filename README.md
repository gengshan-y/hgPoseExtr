# hgPoseExtr
Modified hourglass. Extract pose estimations.

To crop images using detection results, run:

python imgCrop.py batch_size 0-based_taskid

e.g.: python imgCrop.py 10000 0

To run hourglss model on GPU, run

th mulThreadsExtr current_pointer batch_size num_iter outfile_name GPU_num GPU_offset

e.g.:  th mulThreadsExtr.lua 101 5 100 test 4 0

## Notes
Cpu machines crop the images, and GPU machines run neural networks.
25fps on Tesla 40
