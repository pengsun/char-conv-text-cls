--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

--- global config: imdb
--local DATA_PATH = '/home/ps/data/imdb'
--local DATA_OUT = path.join(DATA_PATH, 'word-t7')
--local FN_VOCAB_FREQ = 'imdb_trn-30000.vocab'
--local FN_TOK_TRAIN = 'imdb-train.txt.tok'
--local FN_CAT_TRAIN = 'imdb-train.cat'
--local FN_TOK_TEST = 'imdb-test.txt.tok'
--local FN_CAT_TEST = 'imdb-test.cat'
--local VOCAB_CAT = {pos=1, neg=2}

--- global config: elec 25k, deepml
--local DATA_PATH = '/home/ps/data/elec'
--local DATA_OUT = path.join(DATA_PATH, 'tr25k-word-t7')
--local FN_VOCAB_FREQ = 'elec-25k-train-30000.vocab'
--local FN_TOK_TRAIN = 'elec-25k-train.txt.tok'
--local FN_CAT_TRAIN = 'elec-25k-train.cat'
--local FN_TOK_TEST = 'elec-test.txt.tok'
--local FN_CAT_TEST = 'elec-test.cat'
--local VOCAB_CAT = {['1']=1, ['2']=2}

--- global config: elec 200k, deepml
--local DATA_PATH = '/mnt/data/datasets/Text/elec'
--local DATA_OUT = path.join(DATA_PATH, 'tr200k-word-t7')
--local FN_VOCAB_FREQ = 'elec-200k-train-30000.vocab'
--local FN_TOK_TRAIN = 'elec-200k-train.txt.tok'
--local FN_CAT_TRAIN = 'elec-200k-train.cat'
--local FN_TOK_TEST = 'elec-test.txt.tok'
--local FN_CAT_TEST = 'elec-test.cat'
--local VOCAB_CAT = {['1']=1, ['2']=2}

--- global config: dbpedia
local DATA_PATH = '/home/ps/data/dbpedia'
local DATA_OUT = path.join(DATA_PATH, 'word-t7')
local FN_VOCAB_FREQ = 'tok-cat/train-30000.vocab'
local FN_TOK_TRAIN = 'tok-cat/train.txt.tok'
local FN_CAT_TRAIN = 'tok-cat/train.cat'
local FN_TOK_TEST = 'tok-cat/test.txt.tok'
local FN_CAT_TEST = 'tok-cat/test.cat'
local tmp = function ()
    local cat = {}
    for i = 1, 14 do cat[tostring(i)] = i end
    return cat
end
local VOCAB_CAT = tmp()

-- make vocabulary
local function update_vocab(vocab, str)
    local lines = stringx.splitlines(str)
    for _, line in pairs(lines) do
        for i = 1, #line do
            local c = string.sub(line, i,i)
            if not vocab[c] then
                local count = tablex.size(vocab)
                vocab[c] = count + 1
            end
        end
    end
end

local function read_vocab()
    local vocab = {}

    -- unknown word: always 1
    local unk = '<unknown>'
    vocab[unk] = 1
    local count = 1

    -- scan the file
    local lines = stringx.splitlines( file.read( path.join(DATA_PATH, FN_VOCAB_FREQ) ) )
    for _, line in pairs(lines) do
        local items = stringx.split(line, "\t")
        local word = items[1]

        assert(nil == vocab[word], "should always find new word")
        count = count + 1
        vocab[word] = count
    end

    return vocab
end

-- make dataset
local function line_to_tensor(line, vocab)
    line = string.lower(line) -- to lower case!
    local words = stringx.split(line, ' ')

    local xx = torch.LongTensor(#words):fill(0)
    local numoov = 0
    for i = 1, xx:size(1) do
        local word = words[i]
        if vocab[word] then -- in vocab
            xx[i] = vocab[word]
        else -- oov, always index 1
            xx[i] = 1
            numoov = numoov + 1
        end
    end

    return xx, numoov
end

local function str_to_x_tabletensor(str, vocab)
    local x, numoov = {}, 0
    local lines = stringx.splitlines(str)
    for _, line in pairs(lines) do
        --require'mobdebug'.start()
        local xx, tmp = line_to_tensor(line, vocab)
        table.insert(x, xx)
        numoov = numoov + tmp
    end

    return x, numoov
end

local function str_to_y_tensor(str)
    local lines = stringx.splitlines(str)
    local y = torch.LongTensor(#lines)
    for i = 1, y:numel() do
        local cat = lines[i]
        if VOCAB_CAT[cat] then
            y[i] = VOCAB_CAT[cat]
        else
            error('unknown ' .. cat .. 'at ' .. i)
        end
    end

    return y
end

local function make_t7(fnTok, vocab, fnCat, fnOut)
    print('making t7 dataset...')

    print('reading tokens from ' .. fnTok)
    local strx = file.read(fnTok)
    print('converting...')
    local x, numoov = str_to_x_tabletensor(strx, vocab)
    print('#OOV words = ' .. numoov)

    print('reading cat from ' .. fnCat)
    local stry = file.read(fnCat)
    print('converting...')
    local y = str_to_y_tensor(stry)

    print('saving to ' .. fnOut)
    torch.save(fnOut, {x=x, y=y})
end

local function main()
    -- read and save vocab
    local vocab = read_vocab()
    local fnVocab = path.join(DATA_OUT, 'vocab.t7')
    print('vocab size = ' .. tablex.size(vocab))
    print('saving vocab to ' .. fnVocab)
    torch.save(fnVocab, vocab)

    -- make tr
    make_t7(path.join(DATA_PATH, FN_TOK_TRAIN), vocab,
        path.join(DATA_PATH, FN_CAT_TRAIN),
        path.join(DATA_OUT, 'tr.t7')
    )

    -- make te
    make_t7(path.join(DATA_PATH, FN_TOK_TEST), vocab,
        path.join(DATA_PATH, FN_CAT_TEST),
        path.join(DATA_OUT, 'te.t7')
    )
end

main()