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

local envSavePath = path.join('cv', 'amrevfull-fixtail-word-att')

dofile('train.lua').main{
    envContinuePath = path.join(envSavePath,
        'M225-HU500-KH2KH3KH4-CW9-cvbank.apV4-max-o_epoch7.00_lossval0.8897_errval38.10.t7'
    ),
    envSavePath = envSavePath,
    logSavePath = path.join(envSavePath,
        'M225-HU500-KH2KH3KH4-CW9-cvbank.apV4-max-o_01:36:52@3-30-2016-cont.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}