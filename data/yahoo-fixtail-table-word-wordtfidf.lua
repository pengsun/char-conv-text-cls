if not LoaderTCFixTailWord then
    require'LoaderTCFixTailWord'
end
if not LoaderTCTable then
    require'LoaderTCTable'
end
require'pl.path'

local function ensure_num_words_match(l1, l2)
    print('ensuring #words match...')
    local n = #l1.x
    assert(n == #l2.x)
    for i = 1, n do
        local M1 = l1.x[i]:numel()
        local M2 = l2.x[i]:numel()
        assert(M1==M2, ("doc %d: %d vs %d"):format(i, M1, M2))
    end
end

local this = {}
this.main = function (opt)
    -- options
    --require'mobdebug'.start()
    local opt = opt or {}
    local batSize = opt.batSize or error('no opt.batSize')
    local seqLength = opt.seqLength or error('no opt.seqLength')
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}

    -- data path
    local dataPathWord = "/data/datasets/Text/yahoo-answers/word-t7"
    local dataPathTfidf = "/data/datasets/Text/yahoo-answers/tfidf-t7"

    -- vocab word
    local vocabWord = torch.load(path.join(dataPathWord,'vocab.t7'))
    local unk = 1 -- vocab index for unknown word
    assert(vocabWord['<unknown>']==unk)

    -- arg word
    local argWord = {wordFill = unk}

    -- arg tfidf
    local argTfidf = {wordFill = 0}

    -- tr, val, te data loader
    local tr, val, te

    if dataMask.tr == true then
        print('train data loader')
        -- word loader
        local word = LoaderTCFixTailWord(path.join(dataPathWord, 'tr.t7'),
            batSize, seqLength, argWord
        )
        -- tfidf loader
        local tfidf = LoaderTCFixTailWord(path.join(dataPathTfidf, 'tr.t7'),
            batSize, seqLength, argTfidf
        )
        -- table loader
        ensure_num_words_match(word, tfidf)
        tr = LoaderTCTable(word, tfidf)
        tr:set_order_rand()
        print(tr)
    end

    if dataMask.val == true then
        print('val data loader')
        -- word loader
        local word = LoaderTCFixTailWord(path.join(dataPathWord, 'te.t7'),
            batSize, seqLength, argWord
        )
        -- tfidf loader
        local tfidf = LoaderTCFixTailWord(path.join(dataPathTfidf, 'te.t7'),
            batSize, seqLength, argTfidf
        )
        -- table loader
        ensure_num_words_match(word, tfidf)
        val = LoaderTCTable(word, tfidf)
        val:set_order_natural()
        print(val)
    end

    if dataMask.te == true then
        print('te data loader')
        -- word loader
        local word = LoaderTCFixTailWord(path.join(dataPathWord, 'te.t7'),
            batSize, seqLength, argWord
        )
        -- tfidf loader
        local tfidf = LoaderTCFixTailWord(path.join(dataPathTfidf, 'te.t7'),
            batSize, seqLength, argTfidf
        )
        -- table loader
        ensure_num_words_match(word, tfidf)
        te = LoaderTCTable(word, tfidf)
        te:set_order_natural()
        print(te)
    end

    return tr, val, te, vocabWord
end

return this