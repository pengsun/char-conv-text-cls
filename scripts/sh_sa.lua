--- show saliency

local envPath = 'cv/imdb-fixtailchar'
local envFn = 'seqLength1000-HU256-cv6pl4time3-fc-o_epoch15.00_lossval0.3305.t7'
local seqLength = 1000

--local envPath = 'cv/imdb-randchar'
--local envFn = 'seqLength887-HU1000-cv8max-o_epoch15.00_lossval0.3137.t7'
--local seqLength = 887

math.randomseed(os.time())
local opt = {
    dispWidth = 210,

    dataPath = 'data/imdb-fix.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    batSize = 1,

    ibat = 24833, -- which data batch interested
    iclass = nil, -- which class interested, use that from label y if nil value

    --    ibat = math.random(1,25000), -- which data batch interested
    --    iclass = nil, -- which class interested, use that from label y if nil value
}

dofile('show_saliency.lua').main(opt)
