-- run train.lua
require'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 1e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 200 do
    r[i] = r[i-1] * factor
  end
  return r
end

local function make_lrEpCheckpoint_big()
  local baseRate, factor = 1e-2, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 100 do
    r[i] = r[i-1] * factor
  end
  return r
end

local netname = 'cv24max-o'
local batSize = 250
local seqLength = 887
local HU = 1000

local opt = {
  mdPath = path.join('net', netname .. '.lua'),

  dataPath = 'data/imdb-rand.lua',
  envSavePath = 'cv/imdb-randchar',

  envSavePrefix = 'seqLength' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength,
  V = 81 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 50,

  paramInitBound = 0.01,
  printFreq = 11,
  evalFreq = 300,
--  printFreq = 30,
--  evalFreq = 30,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)