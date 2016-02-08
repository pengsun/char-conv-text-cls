if not LoaderTCFixTailChar then
    require'LoaderTCFixTailChar'
end

local this = {}

this.main = function (opt)
    -- options
    opt = opt or {}
    opt.batSize = opt.batSize or 500
    opt.seqLength = opt.seqLength or 200
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}
    --
    local dataPath = "/mnt/data/datasets/Text/elec/tr200k-char-t7"
    --local dataPath = "/home/ps/data/elec/tr200k-char-t7"

    -- vocabulary
    local v_char = torch.load(path.join(dataPath, 'vocab.t7'))

    -- tr, val, te data loader
    local arg = {charFill = tablex.size(v_char)+1}
    local tr, val, te

    if dataMask.tr == true then
        local fntr = path.join(dataPath, 'tr.t7')
        print('train data loader')
        tr = LoaderTCFixTailChar(fntr, opt.batSize, opt.seqLength, arg)
    end

    if dataMask.val == true then
        local fnval = path.join(dataPath, 'te.t7')
        print('val data loader')
        val = LoaderTCFixTailChar(fnval, opt.batSize, opt.seqLength, arg)
    end

    if dataMask.te == true then
        local fnte = path.join(dataPath, 'te.t7')
        print('test data loader')
        te = LoaderTCFixTailChar(fnte, opt.batSize, opt.seqLength, arg)
    end

    return tr, val, te, v_char
end

return this

