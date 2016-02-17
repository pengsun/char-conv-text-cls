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

local batSize = 125
local trsize = 25*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.021 * itPerEp )
local evalFreq = 1 * itPerEp -- every #epoches

opt = {
    envContinuePath = path.join(
        'cv', 'imdb-fixtail-word',
        'M475-HU500-cv2maxcv3max-o_epoch7.00_lossval0.2126_errval8.20.t7'
    ),
    envSavePath = path.join('cv', 'imdb-fixtail-word'),
    maxEp = 40,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq,
}

dofile('train.lua').main(opt)