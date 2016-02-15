if not LoaderTCFixTailWord then
    require'LoaderTCFixTailWord'
end

local this = {}

this.main = function (opt)
    -- options
    opt = opt or {}
    opt.batSize = opt.batSize or 500
    opt.seqLength = opt.seqLength or 200
    opt.wordLength = opt.wordLength or 12
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}
    --
    --local dataPath = "/mnt/data/datasets/Text/imdb/char2d-t7"
    local dataPath = "/home/ps/data/imdb/char2d-t7"

    -- vocab
    local vocab = torch.load( path.join(dataPath, 'vocab.t7') )

    -- tr, val, te data loader
    local arg = {charFill = assert(vocab[' '], 'no SPACE in vocab')}
    local tr, val, te

    if dataMask.tr == true then
        local fntr = path.join(dataPath, 'tr.t7')
        print('train data loader')
        tr = LoaderTCFixTailWord(fntr, opt.batSize, opt.seqLength, opt.wordLength, arg)
        tr:set_order_rand()
        print(tr)
    end

    if dataMask.val == true then
        local fnval = path.join(dataPath, 'te.t7')
        print('val data loader')
        val = LoaderTCFixTailWord(fnval, opt.batSize, opt.seqLength, opt.wordLength, arg)
        val:set_order_natural()
        print(val)
    end

    if dataMask.te == true then
        local fnte = path.join(dataPath, 'te.t7')
        print('test data loader')
        te = LoaderTCFixTailWord(fnte, opt.batSize, opt.seqLength, opt.wordLength, arg)
        te:set_order_natural()
        print(te)
    end

    return tr, val, te, vocab
end

return this

