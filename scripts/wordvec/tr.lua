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

local netname = 'cp3pl3times3max-o'
local batSize = 250
local seqLength = 297
local HU = 60

local opt = {
  mdPath = path.join('net', 'wordvec', netname .. '.lua'),

  dataPath = 'data/imdb-fixtail-wordvec.lua',
  dataMask = {tr=true,val=true,te=false},

  envSavePath = 'cv/imdb-fixtailwv',
  envSavePrefix = 'seqLength' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength,
  V = 300, --
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 60,

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