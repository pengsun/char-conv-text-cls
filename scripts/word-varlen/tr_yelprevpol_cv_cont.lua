-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 48

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 0.1, 0.1
    local r = {}
    for i = 1, maxEp do
        r[i] = baseRate
    end
    return r
end

local trsize = 560*1000 -- training size
local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches

local envSavePath = path.join('cv-sgd', 'yelprevpol-varlen-word-cont')

dofile('train.lua').main{
    envContinuePath = path.join('cv-sgd', 'yelprevpol-varlen-word',
        'HU500-KH3-cv-max-oV3_epoch24.00_lossval0.1044_errval3.96.t7'
    ),
    envSavePath = envSavePath,
    logSavePath = path.join(envSavePath,
        'HU500-KH3-cv-max-oV3_21:13:00@4-7-2016-cont.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}