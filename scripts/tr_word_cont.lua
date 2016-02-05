-- run train.lua to continue training
require'pl.path'

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

local function make_lrEpCheckpoint_small_tmp()
    local baseRate, factor = 1e-3, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end

    for i = 11, 207 do
        r[i] = r[i-1] * factor
    end
    --- for 208: drop by ten times
    r[208] = r[207] * 0.05
    --- for 209 to rest
    for i = 209, 250 do
        r[i] = r[i-1] * factor
    end

    return r
end

local function make_lrEpCheckpoint_big()
    local baseRate, factor = 2e-2, 0.97
    local r = {}
    for i = 1, 10 do
        r[i] = baseRate
    end
    for i = 11, 250 do
        r[i] = r[i-1] * factor
    end
    return r
end

opt = {
    envContinuePath = path.join(
        'cv', 'imdb-randchar',
        'seqLength887-HU190-cv7-cv5-cv3-fc-o_epoch9.00_lossval0.3663.t7'
    ),
    envSavePath = path.join('cv', 'imdb-randchar'),
    maxEp = 60,
    lrEpCheckpoint = make_lrEpCheckpoint_small(),

    printFreq = 12,
    evalFreq = 300,
}

dofile('train.lua').main(opt)
