--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

--- global config: imdb
--local DATA_PATH = '/home/ps/data/imdb'
--local DATA_OUT = path.join(DATA_PATH, 'char-t7V3')
--local FN_TOK_TRAIN = 'imdb-train.txt.tok'
--local FN_CAT_TRAIN = 'imdb-train.cat'
--local FN_TOK_TEST = 'imdb-test.txt.tok'
--local FN_CAT_TEST = 'imdb-test.cat'
--local VOCAB_CAT = {["pos"]=1, ["neg"]=2}

--- global config: elec 25k
local DATA_PATH = '/home/ps/data/elec'
local DATA_OUT = path.join(DATA_PATH, 'tr25k-char-t7')
local FN_TOK_TRAIN = 'elec-25k-train.txt.tok'
local FN_CAT_TRAIN = 'elec-25k-train.cat'
local FN_TOK_TEST = 'elec-test.txt.tok'
local FN_CAT_TEST = 'elec-test.cat'
local VOCAB_CAT = {["1"]=1, ["2"]=2}

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

local function make_vocab()
    --- from Crepe + SPACE. size 68
    local chars = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{}" .. " "

    local vchar, count = {}, 0
    for i = 1, #chars do
        local c = string.sub(chars, i,i)
        print(i .. ': ' .. c)
        if not vchar[c] then
            count = count + 1
            vchar[c] = count
            print('count ' .. count .. ', inserting ' .. c)
        end
    end

    return vchar
end

-- make dataset
local function line_to_tensor(line, vocab)
    line = string.lower(line) -- to lower case!

    local count = 0
    local xx = torch.ByteTensor(#line)
    for i = 1, xx:numel() do
        local c = string.sub(line, i,i)
        if vocab[c] then -- in vocabulary
            count = count + 1
            xx[count] = vocab[c]
        end
    end

    -- truncate
    xx = xx:resize(count):clone()
    local numoov = #line - count

    return xx, numoov
end

local function str_to_x_tabletensor(str, vocab)
    local x, numoov = {}, 0
    local lines = stringx.splitlines(str)
    for _, line in pairs(lines) do
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

        assert(VOCAB_CAT[cat], "unknown " .. cat .. 'at ' .. i)
        y[i] = VOCAB_CAT[cat]
    end

    return y
end

local function make_t7(fnTok, vocab, fnCat, fnOut)
    print('making t7 dataset...')

    print('reading token from ' .. fnTok)
    local strx = file.read(fnTok)
    print('converting...')
    local x, numoov = str_to_x_tabletensor(strx, vocab)
    print('#OOV chars = ' .. numoov)

    print('reading cat from ' .. fnCat)
    local stry = file.read(fnCat)
    print('converting...')
    local y = str_to_y_tensor(stry)

    print('saving to ' .. fnOut)
    torch.save(fnOut, {x=x, y=y})
end

local function main()
    -- make and save vocab
    local vocab = make_vocab()
    local fnVocab = path.join(DATA_OUT, 'vocab.t7')
    print('made vocab size = ' .. tablex.size(vocab))
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