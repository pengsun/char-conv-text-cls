if not LoaderTCFixTailSentWord then
    require'LoaderTCFixTailSentWord'
end

local this = {}

this.main = function (opt)
    -- options
    opt = opt or {}
    opt.batSize = opt.batSize or error('no opt.batSize')
    opt.numSent = opt.numSent or error('no opt.numSent')
    opt.seqLength = opt.seqLength or error('no opt.seqLength')
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}
    --
    --local dataPath = "/mnt/data/datasets/Text/imdb/googleword2vec-t7"
    local dataPath = "/home/ps/data/elec/tr25k-word-t7"

    -- vocab
    local vocab = torch.load( path.join(dataPath, 'vocab.t7') )
    local unk = 1 -- vocab index for unknown word
    assert(vocab['<unknown>']==unk)

    -- tr, val, te data loader
    local arg = {wordFill = unk,
        sentSym1 = vocab[','],
        sentSym2 = vocab['.'],
        sentSym3 = vocab['!'],
        sentSym4 = vocab['?'],
    }
    local tr, val, te

    if dataMask.tr == true then
        local fntr = path.join(dataPath, 'tr.t7')
        print('train data loader')
        tr = LoaderTCFixTailSentWord(fntr, opt.batSize, opt.numSent, opt.seqLength, arg)
        tr:set_order_rand()
        print(tr)
    end

    if dataMask.val == true then
        local fnval = path.join(dataPath, 'te.t7')
        print('val data loader')
        val = LoaderTCFixTailSentWord(fnval, opt.batSize, opt.numSent, opt.seqLength, arg)
        val:set_order_natural()
        print(val)
    end

    if dataMask.te == true then
        local fnte = path.join(dataPath, 'te.t7')
        print('test data loader')
        te = LoaderTCFixTailSentWord(fnte, opt.batSize, opt.numSent, opt.seqLength, arg)
        te:set_order_natural()
        print(te)
    end

    return tr, val, te, vocab
end

return this

