-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 2e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, 80 do
        r[i] = r[i-1] * factor
    end
    return r
end

local batSize = 125
local trsize = 25*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.061 * itPerEp )
local evalFreq = 3 * itPerEp -- every #epoches

opt = {
    envContinuePath = path.join(
        'cv', 'imdb-fixtail-word-tmp',
        'M475-HU2-cv2max-o_epoch18.00_lossval0.5265_errval20.12.t7'
    ),
    envSavePath = path.join('cv', 'imdb-fixtail-word-tmp'),
    maxEp = 80,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq,
}

dofile('train.lua').main(opt)