require 'image'
require 'paths'
paths.dofile('img.lua')

--[[ split a string to a sequential table on sep ]]--
function strSplit(str, sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    string.gsub(str, pattern, function(c) fields[#fields+1] = c end)
    return fields
end

--[[ shuffle a sequential table t ]]--
function shuffle(t)
  local n = #t
  while n >= 2 do
    local k = math.random(n)
    t[n], t[k] = t[k], t[n]
    n = n - 1
 end
 return t
end

--[[ read line-by-line data to a table, and save total number ]]--
function readDectionList(path, num2Load, ifShuffle)
    num2Load = num2Load or 3000000  -- or load all
    local lines = {}
    local file = io.open(path)
    local it = 0
    for line in file:lines() do
        if it >= num2Load then
            break
        end
        it = it + 1
        table.insert(lines, line)
    end
    file:close()
    print(it .. ' entries loaded')

    if ifShuffle then
        shuffle(lines)
    end

    -- lines['imgNum'] = it

    return lines
end

--[[ get a batch of data ]]--
function getBatch(batch_size)
    for it = currPointerLoc, currPointerLoc + batch_size - 1 do
        local it_mod = (it-1) % batch_size + 1  -- to avoid 0 index

        -- get detection result --
        local detRes = strSplit(detList[it], "\t")

        -- get center
        local center = {}
        table.insert(center, detRes[3])
        table.insert(center, detRes[4])
        centerList[it_mod] = center

        -- get scale
        local scale = detRes[5]
        scaleList[it_mod] = scale

        -- resize img
        local img = detRes[1]
        imgList[it_mod] = img

        timer3 = torch.Timer()
        inpCPU[{{it_mod}, {}, {}, {}}] = crop(image.load(img, 3, 'byte'),
                                              center, scale, 0, 256)
        locLap[3] = locLap[3] + timer3:time().real
    end
    timer3 = torch.Timer()
    inpGPU:copy(inpCPU)
    locLap[4] = locLap[4] + timer3:time().real
end

--[[ Get predicts for the data loaded ]]--
function getPred(batch_size)
    timer3 = torch.Timer()
    inpGPU:mul(1./255)
    locLap[5] = locLap[5] + timer3:time().real

    timer3 = torch.Timer()
    local out = m:forward(inpGPU)
    locLap[6] = locLap[6] + timer3:time().real

    timer3 = torch.Timer()
    hmCPU:copy(out[2])
    locLap[7] = locLap[7] + timer3:time().real

    timer3 = torch.Timer()
    hmCPU[hmCPU:lt(0)] = 0
    locLap[8] = locLap[8] + timer3:time().real
    -- cutorch.synchronize()

    timer3 = torch.Timer()
    for it = 1, batch_size do
        -- get predicted joints positions in cropped and original imgs --
        preds_hm[it],_ = getPreds(hmCPU[it], centerList[it], scaleList[it])
    end
    locLap[9] = locLap[9] + timer3:time().real
end

function dumpResult(batch_size)
    for it = 1, batch_size do
        outFile:write(string.sub(imgList[it], 21, -5), preds_hm[it])
    end
end

function evalBatch(batch_size)
    batch_size = batch_size or batchSize  -- default param is batchSize

    timer2 = torch.Timer()
    getBatch(batch_size)
    locLap[1] = locLap[1] + timer2:time().real
   
    timer2 = torch.Timer()     
    getPred(batch_size)
    locLap[2] = locLap[2] + timer2:time().real

    dumpResult(batch_size)
end

function getPreds(hms, center, scale)
    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(.5)

    -- Get transformed coordinates
    local preds_tf = torch.zeros(preds:size())
    for i = 1,hms:size(1) do        -- Number of samples
        for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
            preds_tf[i][j] = transform(preds[i][j],center,scale,0,hms:size(3),true)
        end
    end

    return preds, preds_tf
end
