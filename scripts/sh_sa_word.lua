--- show saliency
require'cudnn'
require'cunn'
require'onehot-temp-conv'

math.randomseed(os.time())

local envPath = 'cv/elec25k-fixtail-word'
local envFn = 'M275-HU500-cv3maxcv4max-o_epoch6.00_lossval0.2147_errval7.74.t7'
local seqLength = 275

dofile('show_saliency_word.lua').main{
    dataPath = 'data/elec25k-fixtail-word.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    batSize = 1,

    ibat = 19622, -- which data batch interested
    iclass = 2, -- which class interested, use that from label y if nil value

--    ibat = math.random(1,25000), -- which data batch interested
--    iclass = nil, -- which class interested, use that from label y if nil value

    renderer = 'html'
}
