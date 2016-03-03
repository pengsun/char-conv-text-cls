-- run train.lua
require'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 2e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 40 do
    r[i] = r[i-1] * factor
  end
  return r
end

local netname = 'cv2momaxcv3momax-o'
local HU = 500 -- #hidden units
local MO = 2
local seqLength = 225 -- #words per doc

local batSize = 250
local trsize = 560*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.021 * itPerEp )
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local opt = {
    mdPath = path.join('net', 'word', netname .. '.lua'),
    criPath = path.join('net', 'cri-nll-one' .. '.lua'),

    dataPath = 'data/yelprevpol-fixtail-word.lua',
    envSavePath = 'cv/yelprevpol-fixtail-word',

    envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          'MO' .. MO .. '-' ..
          netname,

    seqLength = seqLength, -- #words per doc
    V = 30000 + 1, -- vocab + oov(null)
    HU = HU, -- #hidden units
    MO = MO,
    numClasses = 2, -- #classes

    batSize = batSize,
    maxEp = 30,

    paramInitBound = 0.05,
    printFreq = printFreq,
    evalFreq = evalFreq, -- every #epoches

    lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)