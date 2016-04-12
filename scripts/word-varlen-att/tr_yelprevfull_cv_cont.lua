-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 30

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 0.1, 1
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, maxEp do
        r[i] = baseRate * factor
    end
    return r
end

local trsize = 650*1000 -- training size
local batSize = 100
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches

local envSavePath = path.join('cv-sgd', 'yelprevfull-varlen-word-att')

dofile('train.lua').main{
    envContinuePath = path.join(envSavePath,
        'HU500-KH3-CW9-cv.apV2.3-max-o_epoch24.00_lossval0.8049_errval34.89.t7'
    ),
    envSavePath = path.join('cv-sgd', 'yelprevfull-varlen-word-att-cont'),
    logSavePath = path.join('cv-sgd', 'yelprevfull-varlen-word-att-cont',
        'HU500-KH3-CW9-cv.apV2.3-max-o_13:39:35@4-10-2016-cont.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}