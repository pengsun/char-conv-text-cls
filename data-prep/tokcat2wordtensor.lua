--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

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

local function read_vocab(opt)
    local vocab = {}

    -- unknown word: always 1
    local unk = '<unknown>'
    vocab[unk] = 1
    local count = 1

    -- scan the file
    local lines = stringx.splitlines( file.read( path.join(opt.data_path, opt.fn_vocab_freq) ) )
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

local function str_to_y_tensor(str, vocabCat)
    local lines = stringx.splitlines(str)
    local y = torch.LongTensor(#lines)
    for i = 1, y:numel() do
        local cat = lines[i]
        y[i] = assert(vocabCat[cat],
            'unknown ' .. cat .. 'at ' .. i)
    end

    return y
end

local function make_t7(fnTok, vocab, fnCat, vocabCat, fnOut)
    print('making t7 dataset...')

    print('reading tokens from ' .. fnTok)
    local strx = file.read(fnTok)
    print('converting...')
    local x, numoov = str_to_x_tabletensor(strx, vocab)
    print('#OOV words = ' .. numoov)

    print('reading cat from ' .. fnCat)
    local stry = file.read(fnCat)
    print('converting...')
    local y = str_to_y_tensor(stry, vocabCat)

    print('saving to ' .. fnOut)
    torch.save(fnOut, {x=x, y=y})
end

---
local this = {}
this.main = function(opt)
    -- default/examplar opt
    local get_vocab_cat = function ()
        local cat = {}
        for i = 1, 14 do cat[tostring(i)] = i end
        return cat
    end
    local opt = opt or {
        -- input
        data_path = '/mnt/data/datasets/Text/dbpedia',
        fn_vocab_freq = 'tok-cat/train-30000.vocab',
        vocab_cat = get_vocab_cat(),
        -- output
        data_out = path.join('/mnt/data/datasets/Text/dbpedia', 'word-t7'),
        fn_tok_train = 'tok-cat/train.txt.tok',
        fn_cat_train = 'tok-cat/train.cat',
        fn_tok_test = 'tok-cat/test.txt.tok',
        fn_cat_test = 'tok-cat/test.cat',
    }

    -- read and save vocab
    local vocab = read_vocab(opt)
    local fnVocab = path.join(opt.data_out, 'vocab.t7')
    print('vocab size = ' .. tablex.size(vocab))
    print('saving vocab to ' .. fnVocab)
    torch.save(fnVocab, vocab)

    -- make tr
    make_t7(path.join(opt.data_path, opt.fn_tok_train), vocab,
        path.join(opt.data_path, opt.fn_cat_train), opt.vocab_cat,
        path.join(opt.data_out, 'tr.t7')
    )

    -- make te
    make_t7(path.join(opt.data_path, opt.fn_tok_test), vocab,
        path.join(opt.data_path, opt.fn_cat_test), opt.vocab_cat,
        path.join(opt.data_out, 'te.t7')
    )
end

return this