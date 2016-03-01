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

local netname = 'cv2ap2-cv2max-o'
local batSize = 250
local seqLength = 475
local HU = 256

local trsize = 25*1000
local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.031 * itPerEp )
--local printFreq = 1
local evalFreq = 3 * itPerEp -- every #epoches

local opt = {
  mdPath = path.join('net', 'word', netname .. '.lua'),

  dataPath = 'data/imdb-fixtail-word.lua',
  dataMask = {tr=true, val=true, te=false},

  envSavePath = 'cv/imdb-fixtail-word',
  envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength,
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 40,

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