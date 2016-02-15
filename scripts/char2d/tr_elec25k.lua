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

local netname = 'cv2x3m.cv3x3m.cv4x3m-max-o'
local HU = 300 -- #hidden units
local seqLength = 128 -- #words per doc
local wordLegnth = 12 -- #chars per word

local batSize = 125
local trsize = 25*1000

local itPerEp = math.floor(trsize/batSize)
--local printFreq = math.ceil( 0.061 * itPerEp )
local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local opt = {
  mdPath = path.join('net', 'char2d', netname .. '.lua'),

  dataPath = 'data/elec25k-fixtail-char2d.lua',
  dataMask = {tr = true, val = true, te = false},

  envSavePath = 'cv/elec25k-fixtail-char2d',
  envSavePrefix = 'M' .. seqLength .. '-' ..
          'Q' .. wordLegnth .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength, -- #words per doc
  wordLegnth = wordLegnth, -- #chars per word
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