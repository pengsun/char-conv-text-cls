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

local netname = 'cv3max-max-o'

local batSize = 125
local numSent = 24
local seqLength = 46
local HU = 500

local trsize = 25*1000
local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.061 * itPerEp )
--local printFreq = 1
local evalFreq = 3 * itPerEp -- every #epoches

local opt = {
  mdPath = path.join('net', 'sentword', netname .. '.lua'),

  dataPath = 'data/elec25k-fixtail-sentword.lua',
  dataMask = {tr=true, val=true, te=false},

  envSavePath = 'cv/elec25k-fixtail-sentword',
  envSavePrefix = 'S' .. numSent .. '-' ..
          'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  numSent = numSent,
  seqLength = seqLength,
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 12,

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