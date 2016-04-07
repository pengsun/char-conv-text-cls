-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local maxEp = 6

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

local trsize = 3600*1000
local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local envSavePath = path.join('cv-backup-3-30-2016', 'amrevpol-fixtail-word-att')

dofile('train.lua').main{
    envContinuePath = path.join(envSavePath,
        'M225-HU500-KH3-CW9-cv.apV2-max-o_epoch4.00_lossval0.1478_errval5.32.t7'
    ),
    envSavePath = envSavePath,
    logSavePath = path.join(envSavePath,
        'M225-HU500-KH3-CW9-cv.apV2-max-o_18:22:35@3-21-2016-cont2.log'
    ),

    maxEp = maxEp,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = printFreq,
    evalFreq = evalFreq
}