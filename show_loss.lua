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
    path.join('cv-sgd', 'yelprevfull-varlen-word-wdOutLay1-bat100-V4'),
    path.join('cv-sgd', 'yelprevfull-fixtail-word'),
    path.join('cv-sgd', 'yelprevfull-fixtail-word'),
    path.join('cv-sgd', 'yelprevfull-fixtail-word'),
}
local names = {
    "HU500-KH3-cv-max-oV4_epoch30.00_lossval0.7877_errval34.32",
    "M329-HU500-KH3-cv-max-oV4_epoch30.00_lossval0.7997_errval34.79",
    "M225-HU500-KH3-cv-max-oV4_epoch30.00_lossval0.8089_errval35.05",
    'M118-HU500-KH3-cv-max-oV4_epoch30.00_lossval0.8516_errval37.12',
}
local plot_names = names
local plot_names = {
    'var-len',
    'fix-len-q-90%',
    'fix-len-q-75%',
    'fix-len-q-50%',
}

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
gnuplot.title('val loss')
gnuplot.plot(table.unpack(itemsLossVal))

gnuplot.figure(2)
gnuplot.title('tr loss')
gnuplot.plot(table.unpack(itemsLossTr))

gnuplot.figure(3)
gnuplot.title('yelp-full, test err')
gnuplot.plot(table.unpack(itemsErrVal))

