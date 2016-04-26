-- run train.lua
require 'pl.path'
local timenow = require'util.misc'.get_current_time_str()

local maxEp = 30
local function make_lrEpCheckpoint_small()
  local baseRate, factor = 2e-3, 0.97
  local r = {}
  for i = 1, 3 do
    r[i] = baseRate
  end
  for i = 4, maxEp do
    r[i] = r[i-1] * factor
  end
  return r
end

local dataname = 'yelprevpol-rie-varlen-word'
local numClasses = 2
local trsize = 560*1000

local netname = 'cv.apV2.6-max-o'
local HU = 500
local KH = 3
local CW = 9

local envSavePath = path.join('cv-rmsprop-rie', dataname .. '-att-v2.6')
local envSavePrefix =
        'HU' .. HU .. '-' ..
        'KH' .. KH .. '-' ..
        'CW' .. CW .. '-' ..
        netname
local logSavePath = path.join(envSavePath,
  envSavePrefix ..'_' .. timenow .. '.log'
)

local batSize = 100
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches


dofile('train.lua').main{
  mdPath = path.join('net', 'word-att', netname .. '.lua'),
  criPath = path.join('net', 'cri-nll-one' .. '.lua'),

  dataPath = path.join('data', dataname .. '.lua'),
  dataMask = { tr = true, val = true, te = false },

  envSavePath = envSavePath,
  envSavePrefix = envSavePrefix,
  logSavePath = logSavePath,

  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  KH = KH,
  CW = CW,
  numClasses = numClasses,

  batSize = batSize,
  maxEp = maxEp,
  paramInitBound = 0.01,

  printFreq = printFreq,
  evalFreq = evalFreq, -- every #epoches

  showEpTime = true,
  showIterTime = false,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),

  optimMethod = require'optim'.rmsprop,
  optimState = {
    alpha = 0.95
  },
  weightDecayOutputLayer = 0,
}