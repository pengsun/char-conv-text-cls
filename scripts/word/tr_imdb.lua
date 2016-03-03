-- run train.lua
require 'pl.path'

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 2e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, 40 do
        r[i] = r[i - 1] * factor
    end
    return r
end

local netname = 'cv2max-o'
local batSize = 250
local seqLength = 475
local HU = 2

local trsize = 25 * 1000
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 3 * itPerEp -- every #epoches

local envSavePath = 'cv/imdb-fixtail-word-tmp'
local envSavePrefix = 'M' .. seqLength .. '-' ..
        'HU' .. HU .. '-' ..
        netname

local opt = {
    mdPath = path.join('net', 'word', netname .. '.lua'),
    criPath = path.join('net', 'cri-nll-one' .. '.lua'),
    criWeight = {1.0, 1.0},

    dataPath = 'data/imdb-fixtail-word.lua',
    dataMask = { tr = true, val = true, te = false },

    envSavePath = envSavePath,
    envSavePrefix = envSavePrefix,

    logSavePath = path.join(envSavePath, envSavePrefix .. '.log'),

    seqLength = seqLength,
    V = 30000 + 1, -- vocab + oov(null)
    HU = HU,
    numClasses = 2,

    batSize = batSize,
    maxEp = 18,
    paramInitBound = 0.05,

    printFreq = printFreq,
    evalFreq = evalFreq, -- every #epoches

    showEpTime = true,
    showIterTime = true,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
    learningRate = 2e-3,
    alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)