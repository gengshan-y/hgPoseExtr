--[[ split a string to a sequential table on sep ]]--
function string:split(sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    self:gsub(pattern, function(c) fields[#fields+1] = c end)
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

--[[ -- Test case
a = {}
table.insert(a, 'a')
table.insert(a, 'b')
table.insert(a, 'c')
table.insert(a, 'd')
shuffle(a)
for k, v in pairs(a) do
    print(k .. '  ' .. v)
end]]--

--[[ read line-by-line data to a table, and set the current point to 1 ]]--
function readDectionList(path, num2Load, ifShuffle)
    num2Load = num2Load or 3000000  -- or load all
    local lines ={}
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

    lines['currPointer'] = 1
    return lines
end

--[[ get a batch of data ]]--
function getBatch(batch_size)
    batch_size = batch_size or batchSize  -- default param is batchSize
    for it =detList['currPointer'], detList['currPointer'] + batch_size - 1 do
        local it_mod = (it-1) % batch_size + 1  -- to avoid 0 index
        -- print('mod is ' .. it_mod)
        -- print('count is ' .. it)
        -- get detection result --
        local detRes = string.split(detList[it], "\t")

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
        -- impInp = image.load(img)
        tmpInp = image.load(img, 3, 'byte')

        timer3 = torch.Timer()
        inp = crop(tmpInp, center, scale, 0, 256)
        lap[3] = lap[3] + timer3:time().real

        timer3 = torch.Timer()
        imgBuff[{{it_mod}, {}, {}, {}}] = inp
        lap[4] = lap[4] + timer3:time().real
    end
    detList['currPointer'] = detList['currPointer'] + batch_size
end

--[[ Get predicts for the data loaded ]]--
function getPred(batch_size)
    batch_size = batch_size or batchSize  -- default param is batchSize
    
    timer3 = torch.Timer()
    imgBuff:mul(1./255)
    lap[5] = lap[5] + timer3:time().real

    timer3 = torch.Timer()
    local out = m:forward(imgBuff)
    lap[6] = lap[6] + timer3:time().real

    -- timerx = torch.Timer()
    -- tmpHm = torch.FloatTensor(5, 16, 64, 64):copy(out[2])
    -- tmpHm:copy(torch.rand(5, 16, 64, 64))
    -- print(#tmpHm)
    -- print(torch.type(tmpHm))
    -- print(tmpHm[1])
    -- lap[4] = lap[4] + timerx:time().real

    -- timerx = torch.Timer()
    -- local hm = tmpHm:float()
    timer3 = torch.Timer()
    hmBuff:copy(out[2])
    lap[7] = lap[7] + timer3:time().real
    
    timer3 = torch.Timer()
    hmBuff[hmBuff:lt(0)] = 0
    lap[8] = lap[8] + timer3:time().real
    -- lap[5] = lap[5] + timerx:time().real
    -- hm[hm:lt(0)] = 0
    -- cutorch.synchronize()

    timer3 = torch.Timer()
    for it = 1, batch_size do
        -- get predicted joints positions in cropped and original imgs --
        preds_hm[it], preds_img[it] = getPreds(hmBuff[it], centerList[it], scaleList[it])

        --[[
        preds_hm[it]:mul(4)
        -- raw predicted joints position --
        if  it == 1 then
            print('findished ' .. imgList[it])
            local dispImg = drawOutput(torch.DoubleTensor():resize(imgBuff[it]:size()):copy(imgBuff[it]), hmBuff[it], preds_hm[it][1])
            itorch.image(dispImg)
        end
        --]]
    end
    lap[9] = lap[9] + timer3:time().real

   
end

function dumpResult(batch_size)
    batch_size = batch_size or batchSize  -- default param is batchSize
    for it = 1, batch_size do
        -- file:write(imgList[it], preds_hm[it])
        outFile:write(string.sub(imgList[it], 21, -5), preds_hm[it])
    end
end
