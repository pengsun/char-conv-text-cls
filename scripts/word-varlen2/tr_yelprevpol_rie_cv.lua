-- run train.lua
require 'pl.path'
local timenow = require'util.misc'.get_current_time_str()

local maxEp = 30
local function make_lrEpCheckpoint_small()
  local baseRate, factor = 0.1, 0.1
  local r = {}
  for i = 1, 24 do
    r[i] = baseRate
  end
  for i = 25, maxEp do
    r[i] = baseRate * factor
  end
  return r
end

local dataname = 'yelprevpol-rie-varlen-word'
local numClasses = 2
local trsize = 560*1000

local netname = 'cv-max-oV4.5'
local HU = 500
local KH = 3

local envSavePath = path.join('cv-sgd-rie', dataname..'-wdOutLay1-bat100-lr0.1-oh-v4.5')
local envSavePrefix =
        'HU' .. HU .. '-' ..
        'KH' .. KH .. '-' ..
        netname
local logSavePath = path.join(envSavePath,
  envSavePrefix ..'_' .. timenow .. '.log'
)

local batSize = 100
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches

dofile('train.lua').main{
  mdPath = path.join('net', 'word2', netname .. '.lua'),
  criPath = path.join('net', 'cri-nll-one' .. '.lua'),

  dataPath = path.join('data', dataname .. '.lua'),
  dataMask = { tr = true, val = true, te = false },

  envSavePath = envSavePath,
  envSavePrefix = envSavePrefix,

  logSavePath = logSavePath,

  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  KH = KH,
  numClasses = numClasses,

  batSize = batSize,
  maxEp = maxEp,
  paramInitBound = 0.01,

  printFreq = printFreq,
  evalFreq = evalFreq, -- every #epoches

  showEpTime = true,
  showIterTime = false,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),

  optimMethod = require'optim'.sgd,
  optimState = {
    momentum = 0.9,
    weightDecay = 0,
  },
  weightDecayOutputLayer = 1e-4,
}