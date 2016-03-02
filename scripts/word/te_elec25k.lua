-- run test.lua
require'pl.path'

require'onehot-temp-conv'
require'cunn'
local envFn = [[M275-HU500-cv3maxcv4max-o_epoch24.00_lossval0.3355_errval7.80.t7]]
local envPath = 'cv/elec25k-fixtail-word'
local seqLength = 275

local test = dofile('test.lua')

-- show testing
local outputs, targets = test.main{
    dataPath = 'data/elec25k-fixtail-word.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    --batSize = 7,
    batSize = 250,
}

-- save results
test.save_miscls(outputs, targets,
    {fnMisCls = path.join(envPath, 'M275-HU500-cv3maxcv4max-o_epoch24.00-miscls.txt')}
)