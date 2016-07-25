--[[
 - Filename:        mod_mulThreadsExtr.lua
 - Date:            Jul 25 2016
 - Last Edited by:  Gengshan Yang
 - Description:     mod_mulThreadsExtr current_pointer outfile_name batch_size
                                       GPU_offset
 --]]

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')  -- so that can change global values
local nter = 287600 * 2 -- 2876828 -> 10
local ngpu = 2
local currPointer = torch.IntTensor(1): -- point to current data
                    fill(tonumber(arg[1]))  -- tensor is sharable
local lap = torch.FloatTensor(10):fill(0)  -- to record the time lapse

local pool = threads.Threads(
    ngpu,
    function(thresdid)
        -- necessary dl modules --
        require 'nn'
        require 'nngraph'
        require 'cunn'
        require 'cudnn'
        require 'cutorch'

        -- for extracting pose features --
        require 'mod_mulThreadsExtrUtil'
        require 'hdf5'

        -- paths --
        modelPath = 'umich-stacked-hourglass.t7'
        inputFilePath = '/home/gengshan/workJul/darknet/results/'..
                        'comp4_det_test_person.txt'
        outputFilePath = '/data/gengshan/pose/' ..
                         arg[2] .. thresdid ..'.h5'
        batchSize = arg[3]
    end,
    function(threadid)
        print('starting a new thread# ' .. threadid)
        -- get data
        detList = readDectionList(inputFilePath, false)

        -- open output file
        os.execute('rm ' .. outputFilePath)
        outFile = hdf5.open(outputFilePath, 'a')
 
        -- init models
        cutorch.setDevice(threadid + arg[4])
        m = torch.load(modelPath)
        
        -- init input buffer
        centerList = {}
        scaleList = {}
        imgList = {}
        inpGPU = torch.CudaTensor():resize(batchSize, 3, 256, 256)
        inpCPU = torch.FloatTensor():resize(batchSize, 3, 256, 256)
        
        -- init output buffer
        hmCPU = torch.FloatTensor():resize(batchSize, 16, 64, 64)
        preds_hm = {}

        -- other vars visible to thread, to avoid garbage
        currPointerLoc = 1
    end
)

collectgarbage()
collectgarbage()
local jobdone = 0
local beg = tonumber(os.date"%s")
for it = 1, 100000 do
    pool:addjob(
        function()
            currPointerLoc = currPointer[1]  -- so that funcs in another file can see
            currPointer:add(batchSize)
            print('thread ' .. __threadid .. '. currPointer ' .. currPointerLoc ..
                  ' time ' .. tonumber(os.date"%s") - beg .. 's')

            locLap = torch.FloatTensor(10):fill(0)

            timer2 = torch.Timer()
            getBatch(batchSize)
            locLap[1] = locLap[1] + timer2:time().real

            timer2 = torch.Timer()
            getPred(batch_size)
            locLap[2] = locLap[2] + timer2:time().real

            dumpResult(batch_size)

            timer1 = torch.Timer()
            if it % 5 then
                collectgarbage()
            end
            locLap[10] = locLap[10] + timer1:time().real
            lap:add(locLap)
            return __threadid  -- global var auto-stored when creating threads
        end,

        function(id)
            -- print(string.format("task %d finished (ran on thread ID %x)", i, id))
            -- Clean garbage and close h5 files -- 
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

print(lap[{{1, 2}}])
print(lap[{{3, 4}}])
print(lap[{{5, 9}}])

collectgarbage()
collectgarbage()

pool:terminate()
