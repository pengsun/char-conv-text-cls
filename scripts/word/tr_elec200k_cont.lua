-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 1e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, 40 do
        r[i] = r[i-1] * factor
    end
    return r
end

local batSize = 50

local trsize = 200*1000
local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.061 * itPerEp )
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

opt = {
    envContinuePath = path.join(
        'cv', 'elec200k-fixtail-word',
        'M475-HU750-cv3maxcv4max-o_epoch1.00_lossval0.1895_errval7.46.t7'
    ),
    envSavePath = path.join('cv', 'elec200k-fixtail-word'),
    maxEp = 40,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    printFreq = printFreq,
    evalFreq = evalFreq,
}

dofile('train.lua').main(opt)
