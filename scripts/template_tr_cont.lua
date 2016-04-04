-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 20

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

local trsize = 8848 -- training size
local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local envSavePath = path.join('cv', 'amrevpol-fixtail-word-att')

dofile('train.lua').main{
    envContinuePath = path.join(envSavePath,
        'xxx.t7'
    ),
    envSavePath = envSavePath,
    logSavePath = path.join(envSavePath,
        'xxx-cont.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}