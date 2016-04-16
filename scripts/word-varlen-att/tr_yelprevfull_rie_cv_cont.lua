-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 60

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 0.5, 0.1
    local r = {}
    for i = 1, 24 do
        r[i] = baseRate
    end
    for i = 25, maxEp do
        r[i] = baseRate * factor
    end
    return r
end

local trsize = 650*1000 -- training size
local batSize = 100
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches

local envSavePath = path.join('cv-sgd', 'yelprevfull-varlen-word-att-wdOutLay1-bat100-lr0.5-v2.1')

dofile('train.lua').main{
    envContinuePath = path.join(envSavePath,
        'HU500-KH3-CW9-cv.apV2.1-max-o_epoch30.00_lossval0.7948_errval34.31.t7'
    ),
    envSavePath = envSavePath,
    logSavePath = path.join(envSavePath,
        'HU500-KH3-CW9-cv.apV2.1-max-o_00:27:28@4-11-2016-cont.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}