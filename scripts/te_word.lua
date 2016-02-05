-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M475-HU502-cp3max-o_epoch20.00_lossval0.2221.t7]]
local envPath = 'cv/imdb-fixtail-word'
local seqLength = 475

local opt = {
    dataPath = 'data/imdb-fixtail-word.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    --batSize = 7,
    batSize = 250,
}

local test = dofile('test.lua')
test.main(opt)