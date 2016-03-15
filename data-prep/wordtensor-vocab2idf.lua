--- word tensor, vocab --> tfidf
require'pl.path'
require'pl.stringx'
require'pl.file'
require'pl.tablex'
uv = require'util.vocab'

-- helper
local function assertVocab(v)
    assert(v['<unknown>'] == 1, 'vocab corrupted')
end

local function extract_idf(xx, vocab)

    local function init_vocabDocCount(vocab)
        local vcc = {}
        for word in pairs(vocab) do
            vcc[word] = 0
        end
        return vcc
    end

    local function update_vocabDocCount(vDocCount, doc, iVocab)
        local wordChecked = {}
        for i = 1, doc:numel() do
            local wordIdx = doc[i]
            local word = assert(iVocab[wordIdx], "word indx " .. wordIdx)
            if not wordChecked[word] then
                -- doc count +1
                assert(vDocCount[word], "word " .. word .. "?")
                vDocCount[word] = vDocCount[word] + 1

                -- mark as checked word
                wordChecked[word] = true
            end
        end
    end

    local function check_idf(idf, ndoc)
        local vmax = math.log(ndoc)
        for word, v in pairs(idf) do
            assert(v>=0 and v<=vmax, word .. ": v = " .. v)
        end
    end

    -- vocab to doc
    --require'mobdebug'.start()
    local vocabDocCount = init_vocabDocCount(vocab)
    local iVocab = uv.make_inverse_vocabulary(vocab)
    local ndoc = #xx
    for i = 1, ndoc do
        local doc = xx[i]
        update_vocabDocCount(vocabDocCount, doc, iVocab)
        xlua.progress(i, ndoc)
    end

    -- calculate the idf
    local fun = function (c)
        assert(c>0, "doc count must > 0 ...")
        return math.log( ndoc/c )
    end
    tablex.transform(fun, vocabDocCount)

    -- hack: zero the <unknown> token
    assert(vocabDocCount['<unknown>'])
    vocabDocCount['<unknown>'] = 0

    print('checking idf...')
    local idf = vocabDocCount
    check_idf(idf, ndoc)
    return idf
end

-- exposed
local this = {}
this.main = function(opt)
    -- default/examplar opt
    local dftPath = '/mnt/data/datasets/Text/imdb/word-t7'
    local dftPathTfidf = '/mnt/data/datasets/Text/imdb/tfidf-t7'
    local opt = opt or {
        -- input
        fn_data = path.join(dftPath, 'tr.t7'),
        fn_vocab = path.join(dftPath, 'vocab.t7'),
        -- output
        fn_idf = path.join(dftPathTfidf, 'idf.t7'),
    }

    local data = torch.load(opt.fn_data)
    local xx = data.x
    local vocab = torch.load(opt.fn_vocab)
    assertVocab(vocab)

    print('extracting idf')
    local idf = extract_idf(xx, vocab)

    print('saving to ' .. opt.fn_idf)
    torch.save(opt.fn_idf, idf)
end

return this