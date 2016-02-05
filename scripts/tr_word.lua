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

local netname = 'cv2max3max-o'
local batSize = 125
local seqLength = 475
local HU = 502

local opt = {
  mdPath = path.join('net', 'word', netname .. '.lua'),

  dataPath = 'data/imdb-fixtail-word.lua',
  envSavePath = 'cv/imdb-fixtail-word',

  envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength,
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 20,

  paramInitBound = 0.05,
  printFreq = 31,
  evalFreq = 5 * math.floor(25000/batSize), -- every #epoches
--  printFreq = 30,
--  evalFreq = 30,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)