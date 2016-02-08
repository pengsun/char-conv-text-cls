-- run test.lua
require'pl.path'

local envFn = [[M1014-HU150-cv6-cv4-cv2-fc-o_epoch3.00_lossval0.5538_errval26.71.t7]]
local envPath = 'cv/elec25k-fixtail-char'
local seqLength = 1014

local opt = {
    dataPath = 'data/elec25k-fixtail-char.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    --batSize = 7,
    batSize = 250,
}

local test = dofile('test.lua')
test.main(opt)
