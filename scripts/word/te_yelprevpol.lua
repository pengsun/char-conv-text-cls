-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M225-HU500-cv2maxcv3max-o_epoch6.00_lossval0.1114_errval4.08.t7]]
local envPath = 'cv/yelprevpol-fixtail-word'
local seqLength = 225

local opt = {
    dataPath = 'data/yelprevpol-fixtail-word.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    --batSize = 7,
    batSize = 250,
}

local test = dofile('test.lua')
test.main(opt)