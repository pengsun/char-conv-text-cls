if not LoaderTCFixTailCharWordwin then
    require'LoaderTCFixTailCharWordwin'
end

local this = {}

this.main = function (opt)
    -- options
    opt = opt or {}
    opt.batSize = opt.batSize or 500
    opt.seqLength = opt.seqLength or 200
    opt.winSize = opt.winSize or 12
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}
    --
    --local dataPath = "/mnt/data/datasets/Text/elec/tr25k-char-t7"
    local dataPath = "/data/datasets/Text/elec/tr25k-char-t7"
    --local dataPath = "/home/ps/data/datasets/Text/elec/tr25k-char-t7"

    -- vocab
    local vocab = torch.load( path.join(dataPath, 'vocab.t7') )

    -- tr, val, te data loader
    local arg = {
        charFill = assert(vocab['_'], 'no UNDERLINE in vocab'),
        charWordDlm = assert(vocab[' '], 'no SPACE in vocab'),
    }
    local tr, val, te

    if dataMask.tr == true then
        local fntr = path.join(dataPath, 'tr.t7')
        print('train data loader')
        tr = LoaderTCFixTailCharWordwin(fntr, opt.batSize, opt.seqLength, opt.winSize, arg)
        tr:set_order_rand()
        print(tr)
    end

    if dataMask.val == true then
        local fnval = path.join(dataPath, 'te.t7')
        print('val data loader')
        val = LoaderTCFixTailCharWordwin(fnval, opt.batSize, opt.seqLength, opt.winSize, arg)
        val:set_order_natural()
        print(val)
    end

    if dataMask.te == true then
        local fnte = path.join(dataPath, 'te.t7')
        print('test data loader')
        te = LoaderTCFixTailCharWordwin(fnte, opt.batSize, opt.seqLength, opt.winSize, arg)
        te:set_order_natural()
        print(te)
    end

    return tr, val, te, vocab
end

return this