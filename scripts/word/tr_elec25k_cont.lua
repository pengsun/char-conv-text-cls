-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 2e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, 40 do
        r[i] = r[i-1] * factor
    end
    return r
end

local batSize = 100
local trsize = 25*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.061 * itPerEp )
local evalFreq = 3 * itPerEp -- every #epoches

local timenow = require'util.misc'.get_current_time_str()

opt = {
    envContinuePath = path.join(
        'cv', 'elec25k-fixtail-word',
        'M375-HU500-KH5-MO5-cv-mo-max-o_epoch18.00_lossval0.2255_errval7.39.t7'
    ),
    envSavePath = path.join('cv', 'elec25k-fixtail-word'),
    maxEp = 30,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    logSavePath = path.join(
        'cv', 'elec25k-fixtail-word',
        'M475-HU500-KH5-MO5-cv-mo-max-o_cont_' .. timenow .. '.log'
    ),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq,
}

dofile('train.lua').main(opt)