-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 40

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 0.25, 0.1
    local r = {}
    for i = 1, 24 do
        r[i] = baseRate
    end
    for i = 25, 48 do
        r[i] = baseRate * factor
    end
    for i = 49, maxEp do
        r[i] = baseRate * factor * factor
    end
    return r
end

local trsize = 560*1000 -- training size
local batSize = 100
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches

local envPath = path.join('cv-sgd-rie', 'yelprevpol-rie-varlen-word-wdOutLay1-bat100-lr0.25-att-v2.8')

dofile('train.lua').main{
    envContinuePath = path.join(envPath,
        'HU500-KH3-CW9-cv.apV2.8-max-o_epoch30.00_lossval0.0998_errval3.69.t7'
    ),
    envSavePath = envPath .. '-cont',
    logSavePath = path.join(envPath .. '-cont',
        'HU500-KH3-CW9-cv.apV2.8-max-o_15:46:21@4-26-2016-cont.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}