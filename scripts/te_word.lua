-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M475-HU1000-KH3-cv-max-o_epoch12.00_lossval0.2366_errval8.11.t7]]
local envPath = path.join('cv', 'imdb-fixtail-word')
local seqLength = 475

-- test
local test = dofile('test.lua')
-- display results
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
    {fnMisCls = path.join(envPath, 'M475-HU1000-KH3-cv-max-o_epoch12.00-miscls.txt')}
)