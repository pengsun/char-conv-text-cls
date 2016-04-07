-- run train.lua
require 'pl.path'
local timenow = require'util.misc'.get_current_time_str()

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 2e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 40 do
    r[i] = r[i - 1] * factor
  end
  return r
end

local dataname = 'yelprevfull-varlen-word'
local numClasses = 5
local trsize = 650*1000

local netname = 'cv-max-oV2'
local HU = 500
local KH = 3

local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
local evalFreq = 1 * itPerEp -- every #epoches

local envSavePath = path.join('cv', dataname)
local envSavePrefix =
        'HU' .. HU .. '-' ..
        'KH' .. KH .. '-' ..
        netname
local logSavePath = path.join(envSavePath,
  envSavePrefix ..'_' .. timenow .. '.log'
)

dofile('train.lua').main{
  mdPath = path.join('net', 'word-varlen', netname .. '.lua'),
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
  maxEp = 30,
  paramInitBound = 0.05,

  printFreq = printFreq,
  evalFreq = evalFreq, -- every #epoches

  showEpTime = true,
  showIterTime = true,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),

  optimState = {
    learningRate = 2e-3,
    alpha = 0.95, -- decay rate
  },
}