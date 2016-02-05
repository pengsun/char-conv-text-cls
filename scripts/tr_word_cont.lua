-- run train.lua to continue training
require'pl.path'
require'onehot-temp-conv'

local function make_lrEpCheckpoint_small()
    local baseRate, factor = 1e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, 250 do
        r[i] = r[i-1] * factor
    end
    return r
end

local batSize = 125

opt = {
    envContinuePath = path.join(
        'cv', 'imdb-fixtail-word',
        'M475-HU502-cv2max3max-o_epoch20.00_lossval0.2388.t7'
    ),
    envSavePath = path.join('cv', 'imdb-fixtail-word'),
    maxEp = 25,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    batSize = batSize,
    printFreq = 31,
    evalFreq = 1 * math.floor(25000/batSize), -- every #epoches
}

dofile('train.lua').main(opt)