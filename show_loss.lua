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
require'onehot-temp-conv'
local paths = {
    path.join('cv','imdb-fixtail-word-transfer'),
    --path.join('cv','imdb-fixtail-word'),
}
local names = {
    "M475-HU500-pretr.cv2maxcv3max-o_epoch4.00_lossval0.2619_errval10.70",
    --"M475-HU502-cv2max3max-o_epoch20.00_lossval0.2388",
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

