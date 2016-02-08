-- run train.lua
require'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 1e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 30 do
    r[i] = r[i-1] * factor
  end
  return r
end

local netname = 'cv3maxcv4maxcv5max-o'
local batSize = 50
local seqLength = 475
local HU = 502

local itPerEp = math.floor(25000/batSize)

local opt = {
  mdPath = path.join('net', 'word', netname .. '.lua'),

  dataPath = 'data/elec-fixtail-word.lua',
  envSavePath = 'cv/elec-fixtail-word',

  envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength,
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 41,

  paramInitBound = 0.05,
  printFreq = math.floor(0.11 * itPerEp),
  evalFreq = 3 * itPerEp, -- every #epoches
--  printFreq = 30,
--  evalFreq = 30,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)