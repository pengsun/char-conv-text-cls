-- run train.lua
require'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 2e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 200 do
    r[i] = r[i-1] * factor
  end
  return r
end

local netname = 'cv6-cv4-cv2-fc-o'
local batSize = 250
local seqLength = 1014
local HU = 200

local trsize = 25*1000
local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.11 * itPerEp )
local evalFreq = 3*itPerEp

local opt = {
  mdPath = path.join('net', 'char', netname .. '.lua'),

  dataPath = 'data/elec25k-fixtail-char.lua',
  envSavePath = 'cv/elec25k-fixtail-char',

  envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  seqLength = seqLength,
  V = 68 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 200,

  paramInitBound = 0.05,
  printFreq = printFreq,
  evalFreq = evalFreq,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)