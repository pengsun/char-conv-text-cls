--- show saliency
require'cudnn'
require'cunn'
require'onehot-temp-conv'

math.randomseed(os.time())

local dataPath = 'data/imdb-fixtail-word.lua'
local envPath = 'cv/imdb-fixtail-word-att'
local envFn = 'M475-HU1000-KH3-CW9-cv.ap-max-o_epoch27.00_lossval0.3008_errval7.74.t7'
local seqLength = 475

dofile('show_weight.lua').main{
    dataPath = dataPath,
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    batSize = 1,

    ibat = 418, -- which data batch interested

    --renderer = 'print',
    renderer = 'html',
}