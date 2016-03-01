if not LoaderTCFixTailWord then
    require'LoaderTCFixTailWord'
end
if not LoaderTCFixTailCharWordwin then
    require'LoaderTCFixTailCharWordwin'
end
if not LoaderTCTable then
    require'LoaderTCTable'
end
require'pl.path'

local function ensure_num_words_match(lWord, lChar)
    print('ensuring #words match...')
    local n = #lWord.x
    assert(n == #lChar.x)
    for i = 1, n do
        local M1 = lWord.x[i]:numel()
        local M2 = #lChar.ptrWords[i]
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
    local winSize = opt.winSize or error('no opt.winSize')
    local dataMask = opt.dataMask or {tr=true,val=true,te=true}

    -- data path
    local dataPathWord = "/data/datasets/Text/imdb/word-t7"
    local dataPathChar = "/data/datasets/Text/imdb/char-t7"
    --local dataPathWord = "/home/ps/data/datasets/Text/imdb/word-t7"
    --local dataPathChar = "/home/ps/data/datasets/Text/imdb/char-t7"

    -- vocab word
    local vocabWord = torch.load(path.join(dataPathWord,'vocab.t7'))
    local unk = 1 -- vocab index for unknown word
    assert(vocabWord['<unknown>']==unk)
    local argWord = {wordFill = unk}

    -- vocab char
    local vocabChar = torch.load( path.join(dataPathChar, 'vocab.t7') )
    local argChar = {
        charFill = assert(vocabChar['_'], 'no UNDERLINE in vocab'),
        charWordDlm = assert(vocabChar[' '], 'no SPACE in vocab'),
    }

    -- tr, val, te data loader
    local tr, val, te

    if dataMask.tr == true then
        print('train data loader')
        -- word loader
        local word = LoaderTCFixTailWord(path.join(dataPathWord, 'tr.t7'),
            batSize, seqLength, argWord
        )
        -- char loader
        local char = LoaderTCFixTailCharWordwin(path.join(dataPathChar, 'tr.t7'),
            batSize, seqLength, winSize, argChar
        )
        -- table loader
        ensure_num_words_match(word,char)
        tr = LoaderTCTable(word, char)
        tr:set_order_rand()
        print(tr)
    end

    if dataMask.val == true then
        print('val data loader')
        -- word loader
        local word = LoaderTCFixTailWord(path.join(dataPathWord, 'te.t7'),
            batSize, seqLength, argWord
        )
        -- char loader
        local char = LoaderTCFixTailCharWordwin(path.join(dataPathChar, 'te.t7'),
            batSize, seqLength, winSize, argChar
        )
        -- table loader
        ensure_num_words_match(word,char)
        val = LoaderTCTable(word, char)
        val:set_order_natural()
        print(val)
    end

    if dataMask.te == true then
        print('te data loader')
        -- word loader
        local word = LoaderTCFixTailWord(path.join(dataPathWord, 'te.t7'),
            batSize, seqLength, argWord
        )
        -- char loader
        local char = LoaderTCFixTailCharWordwin(path.join(dataPathChar, 'te.t7'),
            batSize, seqLength, winSize, argChar
        )
        -- table loader
        ensure_num_words_match(word,char)
        te = LoaderTCTable(word, char)
        te:set_order_natural()
        print(te)
    end

    return tr, val, te, vocabWord, vocabChar
end

return this