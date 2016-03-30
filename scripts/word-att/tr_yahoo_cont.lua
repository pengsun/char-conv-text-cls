-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 20

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 2e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, maxEp do
        r[i] = r[i-1] * factor
    end
    return r
end

local trsize = 1400*1000
local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

dofile('train.lua').main{
    envContinuePath = path.join(
        'cv', 'yahoo-fixtail-word-att',
        'M125-HU500-KH2KH3KH4-CW9-cvbank.apV4-max-o_epoch10.00_lossval0.8867_errval26.69.t7'
    ),
    envSavePath = path.join('cv', 'yahoo-fixtail-word-att'),
    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}