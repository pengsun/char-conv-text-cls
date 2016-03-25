-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 15

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 1e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, maxEp do
        r[i] = r[i-1] * factor
    end
    return r
end

local trsize = 3000*1000
local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

dofile('train.lua').main{
    envContinuePath = path.join(
        'cv', 'amrevfull-fixtail-word-att',
        'M225-HU500-KH3-CW9-cv.apV4-max-o_epoch10.00_lossval0.9043_errval38.68.t7'
    ),
    envSavePath = path.join('cv', 'amrevfull-fixtail-word-att'),
    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}