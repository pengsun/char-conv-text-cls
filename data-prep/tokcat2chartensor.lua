--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

-- global
local CHAR_FILL = '_' -- underline for unknown/unll char

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

local function make_vocabChar()
    --- from Crepe + SPACE. size 69
    local chars = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}" .. " "

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
    -- make sure to be consistent with word tensor
    line = string.lower(line) -- to lower case!
    local words = stringx.split(line, ' ')

    local xx = {}
    local numoov = 0

    for i = 1, #words do
        local word = words[i]

        if #word == 0 then -- null word, replace with symbol + SPACE
            table.insert(xx, vocab[CHAR_FILL])
            table.insert(xx, vocab[' '])
        else -- check the word
            for j = 1, #word do -- insert char by char
                local c = word:sub(j,j)
                if vocab[c] then -- in vocab
                    table.insert(xx, vocab[c])
                else -- oov, always fill with char_fill
                    table.insert(xx, vocab[CHAR_FILL])
                    numoov = numoov + 1
                end
            end -- for j

            table.insert(xx, vocab[' ']) -- ended with SPACE
        end -- if #word == 0
    end -- for i

    -- to byte tensor
    local txx = torch.Tensor(xx):byte()
    return txx, numoov
end

--[[
-- obsolete: inconsistent with word tensor
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
]]--

local function str_to_x_tabletensor(str, vocab)
    local x, numoov = {}, 0
    local lines = stringx.splitlines(str)
    for i, line in pairs(lines) do
        local xx, tmp = line_to_tensor(line, vocab)
        table.insert(x, xx)
        numoov = numoov + tmp

        xlua.progress(i, #lines)
    end

    return x, numoov
end

local function str_to_y_tensor(str, vocabCat)
    local lines = stringx.splitlines(str)
    local y = torch.LongTensor(#lines)
    for i = 1, y:numel() do
        local cat = lines[i]

        assert(vocabCat[cat], "unknown " .. cat .. 'at ' .. i)
        y[i] = vocabCat[cat]
    end

    return y
end

local function make_t7(fnTok, vocab, fnCat, vocabCat, fnOut)
    print('making t7 dataset...')

    print('reading token from ' .. fnTok)
    local strx = file.read(fnTok)
    print('converting...')
    local x, numoov = str_to_x_tabletensor(strx, vocab)
    print('#OOV chars = ' .. numoov)

    print('reading cat from ' .. fnCat)
    local stry = file.read(fnCat)
    print('converting...')
    local y = str_to_y_tensor(stry, vocabCat)

    print('saving to ' .. fnOut)
    torch.save(fnOut, {x=x, y=y})
end

-- exposed
local this = {}

this.main = function (opt)

    -- default/examplar opt
    local opt = opt or {
        -- input
        data_path = '/home/ps/data/datasets/Text/elec',
        vocab_cat = {["1"]=1, ["2"]=2},
        fn_tok_train = 'elec-25k-train.txt.tok',
        fn_cat_train = 'elec-25k-train.cat',
        fn_tok_test = 'elec-test.txt.tok',
        fn_cat_test = 'elec-test.cat',
        -- output
        data_out = path.join('/home/ps/data/datasets/Text/elec', 'tr25k-char-t7'),
    }

    -- make and save vocab
    local vocab = make_vocabChar()
    local fnVocab = path.join(opt.data_out, 'vocab.t7')
    print('made vocab size = ' .. tablex.size(vocab))
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