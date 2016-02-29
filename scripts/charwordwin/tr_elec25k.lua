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

local netname = 'cv7x1mp2-cv3x1max-max-o'
local HU = 200 -- #hidden units
local seqLength = 275 -- #words per doc
local winSize = 28 -- #chars per window

local batSize = 100
local trsize = 25*1000

local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.021 * itPerEp )
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local opt = {
  mdPath = path.join('net', 'charwordwin', netname .. '.lua'),

  dataPath = 'data/elec25k-fixtail-charwordwin.lua',
  dataMask = {tr = true, val = true, te = false},

  envSavePath = 'cv/elec25k-fixtail-charwordwin',
  envSavePrefix = 'M' .. seqLength .. '-' ..
          'Q' .. winSize .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength, -- #words per doc
  winSize = winSize, -- #chars per window
  V = 69, -- vocab
  HU = HU, -- #hidden units
  numClasses = 2, -- positive/negative

  batSize = batSize,
  maxEp = 40,

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