-- run test.lua
require'pl.path'

require'onehot-temp-conv'
local envFn = [[M475-HU500-cv2maxcv3max-o_epoch6.00_lossval0.2150_errval8.26.t7]]
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
    {fnMisCls = path.join(envPath, 'M475-HU500-cv2maxcv3max-o_epoch6.00-miscls.txt')}
)