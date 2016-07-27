--[[
 - Filename:        mod_mulThreadsExtr.lua
 - Date:            Jul 25 2016
 - Last Edited by:  Gengshan Yang
 - Description:     Main file for extracting pose features for a list of
 -                  perison detection results
 --]]

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')  -- so that can change global values
local args = arg  -- to pass into threads
local ngpu = tonumber(args[5])
local currPointer = torch.IntTensor(1): -- point to current data
                    fill(tonumber(args[1]))  -- tensor is sharable
local lap = torch.FloatTensor(10):fill(0)  -- to record the time lapse

local pool = threads.Threads(
    ngpu,
    function(threadid)
        print('starting a new thread# ' .. threadid)
        -- necessary dl modules --
        require 'nn'
        require 'nngraph'
        require 'cunn'
        require 'cudnn'
        require 'cutorch'

        -- for extracting pose features --
        require 'mulThreadsExtrUtil'
        require 'hdf5'

        -- paths --
        batchSize = args[2]
        print('batchSize=' .. batchSize)
        modelPath = 'umich-stacked-hourglass.t7'
        inputFilePath = '/home/gengshan/workJul/darknet/results/'..
                        'comp4_det_test_person.txt'
        outputFilePath = '/data/gengshan/pose/' ..
                         args[4] .. threadid ..'.h5'
        print('outfile=' .. outputFilePath)
    end,
    function(threadid)
        -- get data
        detList = readDectionList(inputFilePath, {tonumber(args[1]),
                                                 args[2] * args[3]},
                                  false)
        -- open output file
        os.execute('rm ' .. outputFilePath)
        outFile = hdf5.open(outputFilePath, 'a')
 
        -- init models
        cutorch.setDevice(threadid + args[6])
        print('dev=' .. threadid + args[6])
        m = torch.load(modelPath)
        
        -- init input buffer
        centerList = {}
        scaleList = {}
        imgList = {}
        inpGPU = torch.CudaTensor():resize(batchSize, 3, 256, 256)
        inpCPU = torch.FloatTensor():resize(batchSize, 3, 256, 256)
        
        -- init output buffer
        outGPU = torch.CudaTensor():resize(batchSize, 16, 64, 64)
        hmCPU = torch.FloatTensor():resize(batchSize, 16, 64, 64)
        preds_hm = {}

        -- other vars visible to thread, to avoid garbage
        currPointerLoc = 1
        begLoc = tonumber(args[1])  -- to recover indexing for getBatch()
        timer1 = torch.Timer()  -- for timing in this file
        timer2 = torch.Timer()  -- for timing in Util file
        locLap = torch.FloatTensor(10)  -- syncronize with global lap
        detRes = {}  -- to store detection results
        it_mod = 0  -- number in getBatch for counting
    end
)

collectgarbage()
collectgarbage()
local jobdone = 0
local beg = tonumber(os.date"%s")
print('jobs= ' .. args[3])
for it = 1, args[3] do
    pool:addjob(
        function()
            currPointerLoc = currPointer[1]  -- so that funcs in another file can see
            currPointer:add(batchSize)
            print('thread ' .. __threadid .. '. currPointer ' .. currPointerLoc ..
                  ' time ' .. tonumber(os.date"%s") - beg .. 's')
            locLap:fill(0)

            -- Get a batch of eval data --
            timer1:reset()
            getBatch(batchSize)
            locLap[1] = locLap[1] + timer1:time().real

            -- Get pose estimation --
            timer1:reset()
            getPred(batchSize)
            locLap[2] = locLap[2] + timer1:time().real

            -- Dump results to .h5 file -- 
            dumpResult(batchSize)

            -- collect garbage --
            timer1:reset()
            if it % 100 then
                collectgarbage()
                collectgarbage()
            end
            locLap[10] = locLap[10] + timer1:time().real
            lap:add(locLap)
            return __threadid  -- global var auto-stored when creating threads
        end,

        function(id)
            -- print(string.format("task %d finished (ran on thread ID %x)", i, id))
            jobdone = jobdone + 1
        end
    )
end


pool:specific(true)
for it = 1, ngpu do
    pool:addjob(
        it,
        function()
            outFile:close()
            print('h5_' .. it .. ' closed.')
        end
    )
end

pool:synchronize()

print(string.format("%d jobs done", jobdone))

print('main model:')
print(lap[{{1, 2}}])
print('submodels of model 1:')
print(lap[{{3, 4}}])
print('submodels of model 2:')
print(lap[{{5, 9}}])

collectgarbage()
collectgarbage()

pool:terminate()
