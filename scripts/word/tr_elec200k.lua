-- run train.lua
require'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 1e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 40 do
    r[i] = r[i-1] * factor
  end
  return r
end

local netname = 'cv3maxcv4max-o'
local batSize = 125
local seqLength = 475
local HU = 500

local trsize = 200*1000
local itPerEp = math.floor(trsize/batSize)

local opt = {
  mdPath = path.join('net', 'word', netname .. '.lua'),

  dataPath = 'data/elec200k-fixtail-word.lua',
  envSavePath = 'cv/elec200k-fixtail-word',

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
  printFreq = math.ceil( 0.061 * itPerEp ),
  --printFreq = 1,
  evalFreq = 1 * itPerEp, -- every #epoches

  lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)