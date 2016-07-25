# hgPoseExtr
Modified hourglass. Extract pose estimations.

Usage: th mod_mulThreadsExtr current_pointer batch_size num_iter outfile_name GPU_num GPU_offset

e.g.:  th mod_mulThreadsExtr.lua 101 5 100 test 4 0

## Notes
Cpu machines crop the images, and GPU machines run neural networks.
