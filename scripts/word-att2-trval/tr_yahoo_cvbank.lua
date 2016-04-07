-- run train.lua
require 'pl.path'

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

local function khkh_to_str(khkh)
  local KHSTR = ""
  for i = 1, #khkh do
    KHSTR = KHSTR .. "KH" .. khkh[i]
  end
  return KHSTR
end

local dataname = 'yahoo-fixtail-word-trval'
local numClasses = 10
local trsize = 1400*1000 * 0.9

local netname = 'cvbank.ap-max-o'
local seqLength = 125
local HU = 500
local KHKH = {2, 3, 4}
local CW = 9
local envSavePath = path.join('cv-trval', dataname .. '-att')
local envSavePrefix = 'M' .. seqLength .. '-' ..
        'HU' .. HU .. '-' ..
        khkh_to_str(KHKH) .. '-' ..
        'CW' .. CW .. '-' ..
        netname
local timenow = require'util.misc'.get_current_time_str()
local logSavePath = path.join(envSavePath,
  envSavePrefix ..'_' .. timenow .. '.log'
)

local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches


dofile('train.lua').main{
  mdPath = path.join('net', 'word-att', netname .. '.lua'),
  criPath = path.join('net', 'cri-nll-one' .. '.lua'),

  dataPath = path.join('data', dataname .. '.lua'),
  dataMask = { tr = true, val = true, te = false },

  envSavePath = envSavePath,
  envSavePrefix = envSavePrefix,

  logSavePath = logSavePath,

  seqLength = seqLength,
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  KHKH = KHKH,
  CW = CW,
  numClasses = numClasses,

  batSize = batSize,
  maxEp = 10,
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