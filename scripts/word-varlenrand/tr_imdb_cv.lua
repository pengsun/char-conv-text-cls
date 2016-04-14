-- run train.lua
require'pl.path'
local timenow = require'util.misc'.get_current_time_str()

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 1, 0.1
  local r = {}
  for i = 1, 24 do
    r[i] = baseRate
  end
  for i = 25, 30 do
    r[i] = baseRate * factor
  end
  return r
end

local dataname = 'imdb-varlenrand-word'
local numClasses = 2
local trsize = 25*1000

local netname = 'cv-max-oV4'
local HU = 500
local KH = 3

local envSavePath = path.join('cv-sgd', dataname)
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
local evalFreq = 3 * itPerEp -- every #epoches

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
  maxEp = 30,
  paramInitBound = 0.01,

  printFreq = printFreq,
  evalFreq = evalFreq, -- every #epoches

  showEpTime = true,
  showIterTime = true,
  lrEpCheckpoint = make_lrEpCheckpoint_small(),

  optimMethod = require'optim'.sgd,
  optimState = {
    momentum = 0.9,
    weightDecay = 0,
  },
  weightDecayOutputLayer = 1e-4,
}