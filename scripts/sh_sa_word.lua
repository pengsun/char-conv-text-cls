--- show saliency
require'cudnn'
require'cunn'
require'onehot-temp-conv'

math.randomseed(os.time())

local dataPath = 'data/elec25k-fixtail-word.lua'
local envPath = 'cv/elec25k-fixtail-word'
local envFn = 'M275-HU500-cv3maxcv4max-o_epoch24.00_lossval0.3355_errval7.80.t7'
local seqLength = 275

--local dataPath = 'data/imdb-fixtail-word.lua'
--local envPath = 'cv/imdb-fixtail-word'
--local envFn = 'M475-HU500-cv2maxcv3max-o_epoch6.00_lossval0.2150_errval8.26.t7'
--local seqLength = 475

dofile('show_saliency_word.lua').main{
    dataPath = dataPath,
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    batSize = 1,

    ibat = 22653, -- which data batch interested
    iclass = 1, -- which class interested, use that from label y if nil value

--    ibat = math.random(1,25000), -- which data batch interested
--    iclass = nil, -- which class interested, use that from label y if nil value

    renderer = 'html',
    saType = 'posneg',
}