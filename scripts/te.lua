-- run test.lua
require'pl.path'

local envFn = 'seqLength887-HU190-cv7dr-cv5dr-cv3dr-fc-o_epoch21.00_lossval0.4310.t7'

local opt = {
    dataPath = 'data/imdb-fix.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join('cv', 'imdb-randchar', envFn),

    seqLength = 887,
    --batSize = 7,
    batSize = 250,
}

dofile('test.lua').main(opt)