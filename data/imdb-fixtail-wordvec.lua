if not LoaderTCFixTailWordvec then
    require'LoaderTCFixTailWordvec'
end

local this = {}

this.main = function (opt)
    -- options
    opt = opt or {}
    opt.batSize = opt.batSize or 500
    opt.seqLength = opt.seqLength or 200
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}
    --
    --	local dataPath = "/mnt/data/penn/googleword2vec-t7"
    local dataPath = "/home/ps/data/imdb/googleword2vec-t7"

    -- tr, val, te data loader
    local tr, val, te

    if dataMask.tr == true then
        local fntr = path.join(dataPath, 'tr.t7')
        print('train data loader')
        tr = LoaderTCFixTailWordvec(fntr, opt.batSize, opt.seqLength)
    end

    if dataMask.val == true then
        local fnval = path.join(dataPath, 'te.t7')
        print('val data loader')
        val = LoaderTCFixTailWordvec(fnval, opt.batSize, opt.seqLength)
    end

    if dataMask.te == true then
        local fnte = path.join(dataPath, 'te.t7')
        print('test data loader')
        te = LoaderTCFixTailWordvec(fnte, opt.batSize, opt.seqLength)
    end

    return tr, val, te
end

return this

