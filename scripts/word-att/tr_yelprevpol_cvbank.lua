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

local dataname = 'yelprevpol-fixtail-word'
local numClasses = 2
local trsize = 560*1000

local netname = 'cvbank.ap-max-o'
local seqLength = 225
local HU = 500
local KHKH = {2, 3}
local CW = 9
local envSavePath = path.join('cv', dataname .. '-att')
local envSavePrefix = 'M' .. seqLength .. '-' ..
        'HU' .. HU .. '-' ..
        khkh_to_str(KHKH) .. '-' ..
        'CW' .. CW .. '-' ..
        netname
local timenow = require'util.misc'.get_current_time_str()
local logSavePath = path.join(envSavePath,
  envSavePrefix ..'_' .. timenow .. '.log'
)

local batSize = 100
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