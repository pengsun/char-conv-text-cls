-- run test.lua
require'pl.path'

local envFn = [[
seqLength297-HU60-cp3pl3times3max-o_epoch27.00_lossval0.2668.t7]]
local envPath = 'cv/imdb-fixtailwv'
local seqLength = 297

local opt = {
    dataPath = 'data/imdb-fixtail-wordvec.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    --batSize = 7,
    batSize = 250,
}

local test = dofile('test.lua')
local outputs, targets = test.main(opt)

-- write mis cls
opt.fnMisCls = path.join(envPath, 'miscls-' .. envFn .. '.txt')
test.save_miscls(outputs, targets, opt)