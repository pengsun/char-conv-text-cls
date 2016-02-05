-- run test.lua
require'pl.path'

local envFn = [[
seqLength887-HU1000-cv8max-o_epoch15.00_lossval0.3137.t7]]
local envPath = 'cv/imdb-randchar'

local opt = {
    dataPath = 'data/imdb-fix.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = 887,
    --batSize = 7,
    batSize = 250,
}

local test = dofile('test.lua')
local outputs, targets = test.main(opt)

-- write mis cls
opt.fnMisCls = path.join(envPath, 'miscls-' .. envFn .. '.txt')
test.save_miscls(outputs, targets, opt)