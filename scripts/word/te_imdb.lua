-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M475-HU500-KH3-MO5-cv-mo-max-o_epoch30.00_lossval0.2699_errval7.80.t7]]
local envPath = 'cv/imdb-fixtail-word'
local seqLength = 475

local test = dofile('test.lua')

-- show testing
local outputs, targets = test.main{
    dataPath = 'data/imdb-fixtail-word.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join(envPath, envFn),

    seqLength = seqLength,
    --batSize = 7,
    batSize = 250,
}

-- save results
test.save_miscls(outputs, targets,
    {fnMisCls = path.join(envPath, 'M475-HU500-KH3-MO5-cv-mo-max-o_epoch30.00-miscls.txt')}
)