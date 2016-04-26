require'pl.path'
require'gnuplot'

--- helpers
local function get_errVal(fn)
    require'cudnn'
    require'OneHot'

    local env = torch.load(fn)
    local errVal = env.errVal

    collectgarbage()
    return errVal
end

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

local function get_itValErr_from_file(fn)
    local err = get_errVal(fn)
    local it, val = get_it_val(err)
    return torch.Tensor(it), torch.Tensor(val)
end

local function get_itValLoss_from_file(fn)
    local ell = get_lossesVal(fn)
    local it, val = get_it_val(ell)
    return torch.Tensor(it), torch.Tensor(val)
end

local function get_itTrLoss_from_file(fn)
    local ell = get_lossesTr(fn)
    local it, val = get_it_val(ell)
    return torch.Tensor(it), torch.Tensor(val)
end

--- configs
require'onehot-temp-conv'
local paths = {
    path.join('cv-sgd', 'yahoo-varlen-word-wdOutLay1-bat100-lr0.1-v4'),
    path.join('cv-sgd', 'yahoo-varlen-word-wdOutLay1-bat100-lr0.1-bn'),
}
local names = {
    "HU500-KH3-cv-max-oV4_epoch30.00_lossval0.8314_errval26.48",
    "HU500-KH3-cv-bn-max-o_epoch30.00_lossval0.9451_errval27.33",
}
local plot_names = names
local plot_names = {
    'oh-conv',
    'oh-conv-BN',
}
local figure_name_prefix = 'yahoo'

--- get stuff
local itemsErrVal = {}
local itemsLossVal, itemsLossTr = {}, {}
for i, name in pairs(names) do
    local fn = path.join(paths[i], name .. ".t7")
    local plotName = plot_names[i]

    local itVal, val = get_itValLoss_from_file(fn)
    table.insert(itemsLossVal, {plotName, itVal, val, "~"})

    local itTr, tr = get_itTrLoss_from_file(fn)
    table.insert(itemsLossTr, {plotName, itTr, tr, "~"})

    local itVal, val = get_itValErr_from_file(fn)
    table.insert(itemsErrVal, {plotName, itVal, val, "~"})
end

--- draw
gnuplot.figure(1)
gnuplot.title(figure_name_prefix .. ', val loss')
gnuplot.plot(table.unpack(itemsLossVal))

gnuplot.figure(2)
gnuplot.title(figure_name_prefix .. ', tr loss')
gnuplot.plot(table.unpack(itemsLossTr))

gnuplot.figure(3)
gnuplot.title(figure_name_prefix .. ', test err')
gnuplot.plot(table.unpack(itemsErrVal))

