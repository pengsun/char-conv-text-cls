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

local netname = 'cv2maxcv3maxcv4max-o'
local HU = 500 -- #hidden units
local seqLength = 225 -- #words per doc

local batSize = 250
local trsize = 3600*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.061 * itPerEp )
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local opt = {
  mdPath = path.join('net', 'word', netname .. '.lua'),

  dataPath = 'data/amrevpol-fixtail-word.lua',
  dataMask = {tr=true, val=true, te=false},

  envSavePath = 'cv/amrevpol-fixtail-word',
  envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength, -- #words per doc
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU, -- #hidden units
  numClasses = 2, -- #classes

  batSize = batSize,
  maxEp = 20,

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