--- extract vocabulary
-- TODO: seperate the vocab-truncation code
require'pl.file'
require'pl.stringx'
require'pl.operator'

--- helpers
local function update_vocab(words, v, vc)
    local v = v or error('no vocab')
    local vc = vc or error('no vocab error')

    for i, word in pairs(words) do
        --if i == 9 then require'mobdebug'.start() end

        -- lower case!
        word = string.lower(word)

        -- ignore null word
        if #word > 0 then

            -- in or out vocab?
            if not v[word] then -- insert new word
                assert(nil==vc[word],
                    'vocab and vocabFreq inconsistent, new word = ' .. word
                )
                v[word] = tablex.size(v) + 1
                vc[word] = 1
            else -- in vocab
                assert(vc[word],
                    'vocab and vocabFreq inconsistent, word = ' .. word
                )
                vc[word] = vc[word] + 1
            end
        end
    end
end

local function truncate_vocabFreq(vocabFreq, sz)
    local vt, count = {}, 0
    for word, freq in tablex.sortv(vocabFreq, '>') do
        count = count + 1

        --if count == 2053 then require'mobdebug'.start() end
        if count > sz then break end
        vt[word] = freq
    end
    return vt
end

local function save_vocabFreq_txt(fn, vocabFreq)
    local f = assert(io.open(fn, 'w'))
    for word, freq in tablex.sortv(vocabFreq, '>') do
        f:write(word .. "\t" .. freq .. "\n")
    end
    f:close()
end

---
local this = {}

this.main = function (opt)
    -- default/examplar opt
    local opt = opt or {
        -- input
        fn_tokens = '/mnt/data/datasets/Text/dbpedia/tok-cat/train.txt.tok',
        vocab_truncate_size = 30000,
        -- output
        fn_vocab_freq = '/mnt/data/datasets/Text/dbpedia/tok-cat/train.vocab',
        fn_vocab_freq_truncate = '/mnt/data/datasets/Text/dbpedia/tok-cat/train-30000.vocab',
    }

    -- init vocab
    local vocab, vocabFreq = {}, {}

    -- update vocab for each line in file
    print('extracting vocab from ' .. opt.fn_tokens)
    local lines = stringx.splitlines( file.read(opt.fn_tokens) )
    for i, line in ipairs(lines) do
        --if i == 11696 then require'mobdebug'.start() end

        local words = stringx.split(line, ' ')
        update_vocab(words, vocab, vocabFreq)
        xlua.progress(i, #lines)
    end
    print('#vocab = ' .. tablex.size(vocab))

    -- truncate
    --require'mobdebug'.start()
    print('truncating vocab to size ' .. opt.vocab_truncate_size)
    local vocabFreqTrunc = truncate_vocabFreq(vocabFreq, opt.vocab_truncate_size)
    print('#vocab truncated size = ' .. tablex.size(vocabFreqTrunc))
    assert( tablex.size(vocabFreqTrunc) == opt.vocab_truncate_size )

    -- save
    print('saving to ' .. opt.fn_vocab_freq)
    save_vocabFreq_txt(opt.fn_vocab_freq, vocabFreq)
    print('saving to '.. opt.fn_vocab_freq_truncate)
    save_vocabFreq_txt(opt.fn_vocab_freq_truncate, vocabFreqTrunc)
end

return this
