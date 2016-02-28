--- make 2D char dataset, use pre-defined lower-case word vocab and char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

-- internal config
local MAX_WORD_LENGTH = 12

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

local function read_vocabWord(fnVocabWord)
    local vocab = {}

    -- unknown word: always 1
    local unk = '<unknown>'
    vocab[unk] = 1
    local count = 1

    -- scan the file
    local lines = stringx.splitlines( file.read(fnVocabWord) )
    for _, line in pairs(lines) do
        local items = stringx.split(line, "\t")
        local word = items[1]

        assert(nil == vocab[word], "should always find new word")
        count = count + 1
        vocab[word] = count
    end

    return vocab
end

-- make dataset, helper
local function word_to_chartensor(word, vocabChar)
    local x = torch.ByteTensor(#word)
    for i = 1, #word do
        local c = string.sub(word, i,i)

        -- strict
        --x[i] = assert(vocabChar[c], "out-of-vocab char " .. c .. ', word ' .. word)

        -- loose. substitute with SPACE
        x[i] = vocabChar[c] or vocabChar[' ']
    end
    return x
end

local function line_to_TableTensor(line, vocabWord, vocabChar)
    line = string.lower(line) -- to lower case!
    local words = stringx.split(line, ' ')

    local xx = {}
    local numoov = 0
    for i = 1, #words do
        local word = words[i]
        if vocabWord[word] then -- in vocab word
            xx[i] = word_to_chartensor(word, vocabChar)
        else -- oov word, substitued as a single SPACE
            xx[i] = torch.ByteTensor(1):fill(vocabChar[' '])
            numoov = numoov + 1
        end
    end

    return xx, numoov
end

-- make dataset, helper
local function line_to_tensor2d(line, vocabWord, vocabChar)
    -- lower case words
    line = string.lower(line) -- to lower case!
    local words = stringx.split(line, ' ')

    -- defult xx: filled with SPACE
    local charFill = assert(vocabChar[' '], "no SPACE in vocabChar")
    local xx = torch.ByteTensor(#words, MAX_WORD_LENGTH):fill(charFill)

    -- fill each xx row with the corresponding word
    local numoov = 0
    for i = 1, #words do
        local word = words[i]
        if vocabWord[word] then -- in vocab word
            local tmp = word_to_chartensor(word, vocabChar)
            local realLen = math.min(tmp:numel(), MAX_WORD_LENGTH)
            xx[i][{ {1,realLen} }]:copy( tmp[{ {1,realLen} }] )
        else -- oov word, substitued as a single SPACE
            numoov = numoov + 1
        end
    end

    return xx, numoov
end

-- make dataset
local function str_to_x(str, vocabWord, vocabChar)
    local x, numoov = {}, 0
    local lines = stringx.splitlines(str)
    for i, line in ipairs(lines) do

        -- too slow...
        --local xx, tmp = line_to_TableTensor(line, vocabWord, vocabChar)

        -- should be faster
        local xx, tmp = line_to_tensor2d(line, vocabWord, vocabChar)

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
        y[i] = assert(vocabCat[cat], "unknown " .. cat .. 'at ' .. i)
    end

    return y
end

local function make_t7(fnTok, vocabWord, vocabChar, fnCat, vocabCat, fnOut)
    print('making t7 dataset...')

    print('reading token from ' .. fnTok)
    local strx = file.read(fnTok)
    print('converting...')
    local x, numoov = str_to_x(strx, vocabWord, vocabChar)
    print('#OOV words = ' .. numoov)

    print('reading cat from ' .. fnCat)
    local stry = file.read(fnCat)
    print('converting...')
    local y = str_to_y_tensor(stry, vocabCat)

    assert(#x == y:size(1))
    print('saving to ' .. fnOut)
    torch.save(fnOut, {x=x, y=y})
end

-- exposed
local this = {}

this.main = function (opt)
    -- default/examplar opt
    local opt = opt or {
        -- input
        data_path = '/home/ps/data/elec',
        vocab_cat = {["1"]=1, ["2"]=2},
        -- output
        data_out = path.join('/home/ps/data/elec', 'tr25k-char-t7'),
        fn_tok_train = 'elec-25k-train.txt.tok',
        fn_cat_train = 'elec-25k-train.cat',
        fn_tok_test = 'elec-test.txt.tok',
        fn_cat_test = 'elec-test.cat',
    }

    -- make and save char vocab
    local vocabChar = make_vocabChar()
    local fnVocabChar = path.join(opt.data_out, 'vocab.t7')
    print('made vocab size = ' .. tablex.size(vocabChar))
    print('saving vocab to ' .. fnVocabChar)
    torch.save(fnVocabChar, vocabChar)

    -- read word vocab
    local vocabWord = read_vocabWord(
        path.join(opt.data_path, opt.fn_vocab_freq)
    )

    -- make tr
    make_t7(
        path.join(opt.data_path, opt.fn_tok_train), vocabWord, vocabChar, -- token
        path.join(opt.data_path, opt.fn_cat_train), opt.vocab_cat, -- cat
        path.join(opt.data_out, 'tr.t7') -- out
    )

    -- make te
    make_t7(
        path.join(opt.data_path, opt.fn_tok_test), vocabWord, vocabChar, -- token
        path.join(opt.data_path, opt.fn_cat_test), opt.vocab_cat, -- cat
        path.join(opt.data_out, 'te.t7')
    )
end

return this