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

local batSize = 250
local trsize = 560*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.021 * itPerEp )
local evalFreq = 1 * itPerEp -- every #epoches

opt = {
    envContinuePath = path.join(
        'cv', 'yelprevpol-fixtail-word',
        'M225-HU500-cv2maxcv3max-o_epoch8.00_lossval0.1184_errval4.22.t7'
    ),
    envSavePath = path.join('cv', 'yelprevpol-fixtail-word'),
    maxEp = 40,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq,
}

dofile('train.lua').main(opt)