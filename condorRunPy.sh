#!/bin/bash

source /home/gengshan/workJun/ipynbtest/env2/bin/activate

echo $1
date
echo running as `whoami` on `uname -n` in `pwd`
$1
date
