--- show saliency
require'cudnn'
require'cunn'
require'onehot-temp-conv'

math.randomseed(os.time())

--local dataPath = 'data/elec25k-fixtail-word.lua'
--local envPath = 'cv/elec25k-fixtail-word'
--local envFn = 'M275-HU500-cv3maxcv4max-o_epoch24.00_lossval0.3355_errval7.80.t7'
--local seqLength = 275

local dataPath = 'data/imdb-fixtail-word.lua'
local envPath = 'cv/imdb-fixtail-word-tmp'

--local envFn = 'M475-HU500-KH5-MO5-cv-mo-max-o_epoch21.00_lossval0.2511_errval7.85.t7'
--local envFn = 'M475-HU500-KH5-cv-max-o_epoch21.00_lossval0.2959_errval8.61.t7'

--local envFn = 'M475-HU250-KH2KH3-cvbank-max-o_epoch18.00_lossval0.2469_errval8.30.t7'
--local envFn = 'M475-HU500-KH3-cv-max-o_epoch12.00_lossval0.2236_errval8.36.t7'
local envFn = 'M475-HU500-KH3-MO5-cv-mo-max-o_epoch30.00_lossval0.2699_errval7.80.t7'

local seqLength = 475

dofile('show_saliency_word.lua').main{
    dataPath = dataPath,
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    batSize = 1,

    ibat = 22287, -- which data batch interested
    iclass = 1, -- which class interested, use that from label y if nil value

--    ibat = math.random(1,25000), -- which data batch interested
--    iclass = nil, -- which class interested, use that from label y if nil value

    renderer = 'html',
    saType = 'pos',
}