-- run train.lua
require 'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 5e-3, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 40 do
    r[i] = r[i - 1] * factor
  end
  return r
end

local dataname = 'imdb-fixtail-table-word-wordtfidf'
local numClasses = 2
local trsize = 25*1000

local netname = 'cv.tfidf-max-o'
local seqLength = 475
local HU = 1000
local KH = 3
local envSavePath = path.join('cv', dataname)
local envSavePrefix = 'M' .. seqLength .. '-' ..
        'HU' .. HU .. '-' ..
        'KH' .. KH .. '-' ..
        netname
local timenow = require'util.misc'.get_current_time_str()
local logSavePath = path.join(envSavePath,
  envSavePrefix ..'_' .. timenow .. '.log'
)

local batSize = 250
local itPerEp = math.floor(trsize / batSize)
local printFreq = math.ceil(0.061 * itPerEp)
--local printFreq = 1
local evalFreq = 3 * itPerEp -- every #epoches


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
  KH = KH,
  numClasses = numClasses,

  batSize = batSize,
  maxEp = 90,
  paramInitBound = 0.02,

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