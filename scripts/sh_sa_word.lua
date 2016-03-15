--- show saliency
require'cudnn'
require'cunn'
require'onehot-temp-conv'

math.randomseed(os.time())

local dataPath = 'data/imdb-fixtail-word.lua'
local envPath = 'cv/imdb-fixtail-word'
local envFn = 'M475-HU1000-KH3-cv-max-o_epoch12.00_lossval0.2366_errval8.11.t7'
local seqLength = 475

--local dataPath = 'data/imdb-fixtail-word.lua'
--local envPath = 'cv/imdb-fixtail-word-att'
--local envFn = 'M475-HU1000-KH3-CW9-cv.ap-max-o_epoch21.00_lossval0.2665_errval7.87.t7'
--local seqLength = 475

dofile('show_saliency_word.lua').main{
    dataPath = dataPath,
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    batSize = 1,

    ibat = 24568, -- which data batch interested
    iclass = 1, -- which class interested, use that from label y if nil value

--    ibat = math.random(1,25000), -- which data batch interested
--    iclass = nil, -- which class interested, use that from label y if nil value

    renderer = 'html',
    saType = 'pos',
}