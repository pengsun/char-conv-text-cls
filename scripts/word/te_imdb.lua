-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M475-HU500-KH5-cv-max-o_epoch21.00_lossval0.2959_errval8.61.t7]]
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
    batSize = 100,
}

-- save results
test.save_miscls(outputs, targets,
    {fnMisCls = path.join(envPath, 'M475-HU500-KH5-cv-max-o_epoch21.00-miscls.txt')}
)