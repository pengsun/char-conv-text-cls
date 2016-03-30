--- subsample a set. e.g., make a tr and val set by random sampling
require'pl.path'
require'pl.stringx'
require'pl.file'

local function random_sample(num, numSample)
    assert(numSample < num)
    local tmp = torch.randperm(num)
    local indSample = tmp[{ {1, numSample} }]:totable()
    local indRemain = tmp[{ {numSample+1, -1} }]:totable()
    return indSample, indRemain
end

local function sample_x(x, ind)
    local xx = {}
    for _, i in ipairs(ind) do
        table.insert(xx, x[i]:clone())
    end
    return xx
end

local function sample_y(y, ind)
    local tind = torch.LongTensor(ind)
    local yy = y:index(1, tind):clone()
    return yy
end

local function sample_data(d, ind)
    local xx = sample_x(d.x, ind)
    local yy = sample_y(d.y, ind)
    local dd = {x = xx, y = yy }
    return dd
end

-- exposed
local this = {}
this.main = function(opt)
    -- default/examplar opt
    local opt = opt or {
        -- input
        fn_data = 'tr.t7',
        num_sample = 100,
        -- output
        fn_sample = 'trval.t7',
        fn_remain = 'trtr.t7',
    }

    -- original data
    --require'mobdebug'.start()
    print('loading tr from ' .. opt.fn_data)
    local d = torch.load(opt.fn_data)
    local num = #d.x
    assert(num == d.y:numel())

    -- sub sample
    local indSample, indRemain = random_sample(num, opt.num_sample)
    local dSample = sample_data(d, indSample)
    local dRemain = sample_data(d, indRemain)

    -- save
    print('saving to ' .. opt.fn_sample)
    torch.save(opt.fn_sample, dSample)
    print('saving to ' .. opt.fn_remain)
    torch.save(opt.fn_remain, dRemain)
end

return this