--- word tensor, vocab --> tfidf
require'pl.path'
require'pl.stringx'
require'pl.file'
uv = require'util.vocab'

-- helper
local function assertVocab(v)
    assert(v['<unknown>'] == 1, 'vocab corrupted')
end

local function doc_to_tf(doc)
    -- first pass: word count
    local wordIdxCount = {}
    for i = 1, doc:numel() do
        local wordIdx = doc[i]
        wordIdxCount[wordIdx] = wordIdxCount[wordIdx] or 0
        wordIdxCount[wordIdx] = wordIdxCount[wordIdx] + 1
    end

    -- second pass: tf for each word
    local nword = doc:numel(); assert(nword>=1);
    local doc_tf = {}
    for i = 1, nword do
        local wordIdx = doc[i]
        local count = wordIdxCount[wordIdx] or error("wordIdx " .. wordIdx)
        assert(count >= 1)
        local freq = count/nword
        table.insert(doc_tf, freq)
    end

    return doc_tf
end

local function to_tfidf(doc, tf, idf, ivocab)
    local tfidf = {}
    for i = 1, doc:numel() do
        local wordIdx = doc[i]
        local word = ivocab[wordIdx]
        local iDocFreq = assert(idf[word])

        local tFreq = tf[i]

        tfidf[i] = iDocFreq * tFreq
    end
    return tfidf
end

local function extract_tfidf(x, vocab, idf)
    local tfidf = {}
    local ivocab = uv.make_inverse_vocabulary(vocab)

    for i = 1, #x do
        -- doc to tf for each word
        local doc = x[i]
        local tf = doc_to_tf(doc)

        -- tf, idf -> tfidf
        local doc_tfidf = to_tfidf(doc, tf, idf, ivocab)

        -- to tensor
        local t = torch.FloatTensor(doc_tfidf)

        -- normalize each word's tfidf to [0, 1] by linear mappling
        local function normalize(t)
            local maxValue = t:max()
            t:apply(function (v) return v/maxValue end)
        end
        normalize(t)

        table.insert(tfidf, t)
        xlua.progress(i, #x)
    end

    return tfidf
end

-- exposed
local this = {}
this.main = function(opt)
    -- default/examplar opt
    local dftPath = '/mnt/data/datasets/Text/imdb/word-t7'
    local dftPathTfidf = '/mnt/data/datasets/Text/imdb/tfidf-t7'
    local opt = opt or {
        -- input
        fn_data = path.join(dftPath, 'te.t7'),
        fn_vocab = path.join(dftPath, 'vocab.t7'),
        fn_idf = path.join(dftPathTfidf, 'idf.t7'),
        -- output
        fn_tfidf = path.join(dftPathTfidf, 'te.t7'),
    }

    --require'mobdebug'.start()
    local data = torch.load(opt.fn_data)
    local x, y = data.x, data.y
    local vocab = torch.load(opt.fn_vocab)
    assertVocab(vocab)
    local idf = torch.load(opt.fn_idf)

    print('extracting tfidf')
    local tfidf = extract_tfidf(x, vocab, idf)

    print('saving to ' .. opt.fn_tfidf)
    torch.save(opt.fn_tfidf, {x = tfidf, y = y})
end

return this