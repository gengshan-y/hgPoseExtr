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
function readDectionList(path, range2Load, ifShuffle)
    range2Load = range2Load or {1, 3000000}  -- if nil then or load all
    local lines = {}
    local file = io.open(path)
    local it = 1
    for line in file:lines() do
        if it >= range2Load[1] and it < range2Load[1] + range2Load[2] then
            table.insert(lines, line)
        end
        it = it + 1
    end
    file:close()
    print(range2Load[2] .. ' entries loaded')

    if ifShuffle then
        shuffle(lines)
    end

    return lines
end

--[[ get a batch of data ]]--
function getBatch(batch_size)
    for it = currPointerLoc, currPointerLoc + batch_size - 1 do
        it_mod = (it-1) % batch_size + 1  -- to avoid 0 index

        -- get a table of results --
        detRes = strSplit(detList[it - begLoc + 1], "\t")  -- to subtract offset

        -- get center
        centerList[it_mod] = {detRes[3], detRes[4]}

        -- get scale
        scaleList[it_mod] = detRes[5]

        -- resize img
        imgList[it_mod] = detRes[1]:gsub("tmp", "poseTmp")  -- separated version

        timer2:reset()
        inpCPU[{{it_mod}, {}, {}, {}}] = image.load(imgList[it_mod], 3, 'byte')
        locLap[3] = locLap[3] + timer2:time().real
    end

    timer2:reset()
    inpGPU:copy(inpCPU)
    locLap[4] = locLap[4] + timer2:time().real
end

--[[ Get predicts for the data loaded ]]--
function getPred(batch_size)
    batch_size = batch_size or batchSize  -- default param is batchSize

    timer2:reset()
    inpGPU:mul(1./255)
    locLap[5] = locLap[5] + timer2:time().real

    timer2:reset()
    outGPU = m:forward(inpGPU)[2]
    locLap[6] = locLap[6] + timer2:time().real

    timer2:reset()
    hmCPU:copy(outGPU)
    locLap[7] = locLap[7] + timer2:time().real

    timer2:reset()
    hmCPU[hmCPU:lt(0)] = 0
    locLap[8] = locLap[8] + timer2:time().real

    timer2:reset()
    for it = 1, batch_size do
        -- get predicted joints positions in cropped and original imgs --
        preds_hm[it],_ = getPreds(hmCPU[it], centerList[it], scaleList[it])
    end
    locLap[9] = locLap[9] + timer2:time().real
end

function dumpResult(batch_size)
    batch_size = batch_size or batchSize  -- default param is batchSize
    for it = 1, batch_size do
        outFile:write(string.sub(imgList[it], 24, -5), preds_hm[it])
    end
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
