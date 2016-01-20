require'pl.path'
require'gnuplot'


local function get_lossesVal(fn)
    require'cudnn'
    require'OneHot'

    local env = torch.load(fn)
    local lossesVal = env.lossesVal

    collectgarbage()
    return lossesVal
end

local function get_lossesTr(fn)
    require'cudnn'
    require'OneHot'

    local env = torch.load(fn)
    local lossesTr = env.lossesTr

    collectgarbage()
    return lossesTr
end

local function get_it_val(ell)
    local it = tablex.keys(ell)
    local val = tablex.values(ell)
    return it, val
end

local function get_it_val_from_file(fn)
    local ell = get_lossesVal(fn)
    local it, val = get_it_val(ell)
    return torch.Tensor(it), torch.Tensor(val)
end

local function get_it_tr_from_file(fn)
    local ell = get_lossesTr(fn)
    local it, val = get_it_val(ell)
    return torch.Tensor(it), torch.Tensor(val)
end

---[[]]
local paths = {
    path.join('cv','imdb-randchar', 'tmp'),
    path.join('cv','imdb-randchar', 'tmp2'),
}
local names = {
    "seqLength887-HU190-cv7-cv5-cv3-fc-o_epoch15.00_lossval0.4645",
    "seqLength887-HU190-cv7-cv5-cv3-fc-o_epoch18.00_lossval0.6805",
}


local itemsVal, itemsTr = {}, {}
for i, name in pairs(names) do
    local fn = path.join(paths[i], name .. ".t7")

    local itVal, val = get_it_val_from_file(fn)
    table.insert(itemsVal, {name, itVal, val, "~"})

    local itTr, tr = get_it_tr_from_file(fn)
    table.insert(itemsTr, {name, itTr, tr, "~"})
end

gnuplot.figure(1)
gnuplot.title('val loss')
gnuplot.plot(table.unpack(itemsVal))

gnuplot.figure(2)
gnuplot.title('tr loss')
gnuplot.plot(table.unpack(itemsTr))

