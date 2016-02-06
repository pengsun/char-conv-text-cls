require'pl.file'
require'pl.stringx'
require'pl.operator'

--- global config: elec
local FN_WORDS = '/home/ps/data/elec/elec-25k-train.txt.tok'
local FN_VOCAB_FREQ = '/home/ps/data/elec/elec-25k-train.vocab'

local VOCAB_TRUNCATE_SIZE = 30000
local FN_VOCAB_FREQ_TRUNCATE = '/home/ps/data/elec/elec-25k-train-30000.vocab'

--- helpers
local function update_vocab(words, v, vc)
    local v = v or error('no vocab')
    local vc = vc or error('no vocab error')

    for _, word in pairs(words) do
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
        if count > sz then break end
        vt[word] = freq
    end
    return vt
end

local function save_vocabFreq_txt(fn, vocabFreq)
    local f = assert(io.open(fn, 'w'))
    for word, freq in tablex.sortv(vocabFreq, '>') do
        local str = ("%s\t%s"):format(word, freq)
        f:write(str .. "\n")
    end
    f:close()
end

local function main()
    -- init vocab
    local vocab, vocabFreq = {}, {}

    -- update vocab for each line in file
    print('extracting vocab from ' .. FN_WORDS)
    local lines = stringx.splitlines( file.read(FN_WORDS) )
    for _, line in ipairs(lines) do
        local words = stringx.split(line, ' ')
        update_vocab(words, vocab, vocabFreq)
    end
    print('#vocab = ' .. tablex.size(vocab))

    -- truncate
    --require'mobdebug'.start()
    print('truncating vocab to size ' .. VOCAB_TRUNCATE_SIZE)
    local vocabFreqTrunc = truncate_vocabFreq(vocabFreq, VOCAB_TRUNCATE_SIZE)
    print('#vocab truncated size = ' .. tablex.size(vocabFreqTrunc))
    assert( tablex.size(vocabFreqTrunc) == VOCAB_TRUNCATE_SIZE )

    -- save
    print('saving to ' .. FN_VOCAB_FREQ)
    save_vocabFreq_txt(FN_VOCAB_FREQ, vocabFreq)
    print('saving to '.. FN_VOCAB_FREQ_TRUNCATE)
    save_vocabFreq_txt(FN_VOCAB_FREQ_TRUNCATE, vocabFreqTrunc)
end

main()
