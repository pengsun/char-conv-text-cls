-- run train.lua, no iteration, just duplicate that model
require'pl.path'

local function make_lrEpCheckpoint_small()
  local baseRate, factor = 0, 0.97
  local r = {}
  for i = 1, 10 do
    r[i] = baseRate
  end
  for i = 11, 40 do
    r[i] = r[i-1] * factor
  end
  return r
end

local netname = 'pretrcv2maxcv3max-o'
local batSize = 125
local seqLength = 475
local HU = 500

local trsize = 25*1000
local itPerEp = math.floor(trsize/batSize)
local printFreq = math.ceil( 0.071 * itPerEp )
--local printFreq = 1
local evalFreq = 1 * itPerEp -- every #epoches

local datasetFolder = '/home/ps/data'
local fnVocabThis = path.join(datasetFolder, 'imdb', 'word-t7', 'vocab.t7')
local fnVocabThat = path.join(datasetFolder, 'yelp-review-polarity', 'word-t7', 'vocab.t7')
local fnEnvThat = path.join('cv/yelprevpol-fixtail-word',
  'M225-HU500-cv2maxcv3max-o_epoch25.00_lossval0.1177_errval4.03.t7'
)

local opt = {
  mdPath = path.join('net', 'word-transfer', netname .. '.lua'),

  dataPath = 'data/imdb-fixtail-word.lua',
  dataMask = {tr=true, val=true, te=false},

  envSavePath = 'cv/imdb-fixtail-word-transfer',
  envSavePrefix = 'M' .. seqLength .. '-' ..
          'HU' .. HU .. '-' ..
          netname,

  fnVocabThis = fnVocabThis,
  fnVocabThat = fnVocabThat,
  fnEnvThat = fnEnvThat,

  seqLength = seqLength,
  V = 30000 + 1, -- vocab + oov(null)
  HU = HU,
  numClasses = 2,

  batSize = batSize,
  maxEp = 1,

  paramInitBound = 0.0,
  printFreq = 1,
  evalFreq = 1,

  lrEpCheckpoint = make_lrEpCheckpoint_small(),
}

opt.optimState = {
  learningRate = 2e-3,
  alpha = 0.95, -- decay rate
}

dofile('train.lua').main(opt)