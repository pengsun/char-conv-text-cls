if not LoaderTCVarLenRandWord then
    require'LoaderTCVarLenRandWord'
end

local this = {}

this.main = function (opt)
    -- options
    opt = opt or {}
    opt.batSize = opt.batSize or error('no opt.batSize')
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}
    --
    --local dataPath = "/mnt/data/datasets/Text/yelp-review-full/word-t7-rie"
    local dataPath = "/data/datasets/Text/yelp-review-full/word-t7-rie"
    --local dataPath = "/home/ps/data/yelp-review-full/word-t7-rie"

    -- vocab
    local vocab = torch.load( path.join(dataPath, 'vocab.t7') )
    local unk = 1 -- vocab index for unknown word
    assert(vocab['<unknown>']==unk)

    -- tr, val, te data loader
    local arg = {wordFill = unk}
    local tr, val, te

    if dataMask.tr == true then
        local fntr = path.join(dataPath, 'tr.t7')
        print('train data loader')
        tr = LoaderTCVarLenRandWord(fntr, opt.batSize, arg)
        tr:set_order_rand()
        print(tr)
    end

    if dataMask.val == true then
        local fnval = path.join(dataPath, 'te.t7')
        print('val data loader')
        val = LoaderTCVarLenRandWord(fnval, opt.batSize, arg)
        val:set_order_natural()
        print(val)
    end

    if dataMask.te == true then
        local fnte = path.join(dataPath, 'te.t7')
        print('test data loader')
        te = LoaderTCVarLenRandWord(fnte, opt.batSize, arg)
        te:set_order_natural()
        print(te)
    end

    return tr, val, te, vocab
end

return this

