-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M475-HU1000-KH3-CW9-cv.ap-max-o_epoch27.00_lossval0.3008_errval7.74.t7]]
local envPath = path.join('cv', 'imdb-fixtail-word-att')
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
    {fnMisCls = path.join(envPath, 'M475-HU1000-KH3-CW9-cv.ap-max-o_epoch27.00-miscls.txt')}
)