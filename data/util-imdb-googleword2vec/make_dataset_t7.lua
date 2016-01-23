--- xx: N, {Mi, 300} yy: N
require'pl.path'
require'pl.stringx'
require'pl.file'
require'xlua'

--local DATA_PATH = '/home/ps/data/imdb'
local DATA_PATH = '/mnt/data/datasets/Text/imdb'
local DATA_OUT = path.join(DATA_PATH, 'googleword2vec-t7')
local FN_TOK_TRAIN = 'imdb-train.txt.tok'
local FN_CAT_TRAIN = 'imdb-train.cat'
local FN_TOK_TEST = 'imdb-test.txt.tok'
local FN_CAT_TEST = 'imdb-test.cat'

local function line_to_tensor(line, wv)
    --line = string.lower(line) -- to lower case!
    local words = stringx.split(line)

    local VECSIZE = 300
    local xx = torch.FloatTensor(#words, VECSIZE)
    local count = 0
    for i = 1, xx:numel() do
        local w = words[i]
        local vec = wv:word2vec(w)
        if vec then -- in vocabulary
            count = count + 1
            xx[count] = vec:clone()
        end
    end

    -- truncate
    xx = xx:resize(count, VECSIZE):clone()
    local numoov = #words - count

    return xx, numoov
end

local function str_to_x_tabletensor(str, wv)
    local x, numoov = {}, 0
    local lines = stringx.splitlines(str)
    for i, line in pairs(lines) do
        local xx, tmp = line_to_tensor(line, wv)
        table.insert(x, xx)
        numoov = numoov + tmp

        xlua.progress(i, #lines)
    end

    return x, numoov
end

local function str_to_y_tensor(str)
    local lines = stringx.splitlines(str)
    local y = torch.LongTensor(#lines)
    for i = 1, y:numel() do
        local cat = lines[i]
        if cat == "pos" then
            y[i] = 2
        elseif cat == "neg" then
            y[i] = 1
        else
            error('unknown ' .. cat .. 'at ' .. i)
        end
    end

    return y
end

local function make_t7(fnTok, wv, fnCat, fnOut)
    print('making t7 dataset...')

    print('reading token from ' .. fnTok)
    local strx = file.read(fnTok)
    print('converting...')
    local x, numoov = str_to_x_tabletensor(strx, wv)
    print('#OOV words = ' .. numoov)
    local function get_avg_len(x)
        local count = 0
        for i = 1, #x do
            count = count + x[i]:size(1)
        end
        return count
    end
    print('#avg words = ' .. get_avg_len(x)/#x)

    print('reading cat from ' .. fnCat)
    local stry = file.read(fnCat)
    print('converting...')
    local y = str_to_y_tensor(stry)

    print('saving to ' .. fnOut)
    torch.save(fnOut, {x=x, y=y})
end

local function main()
    -- word2vec handle
    local wv = require'util.googleword2vec.w2vutils'

    -- make tr
    make_t7(path.join(DATA_PATH, FN_TOK_TRAIN), wv,
        path.join(DATA_PATH, FN_CAT_TRAIN),
        path.join(DATA_OUT, 'tr.t7')
    )
    collectgarbage()

    -- make te
--    make_t7(path.join(DATA_PATH, FN_TOK_TEST), wv,
--        path.join(DATA_PATH, FN_CAT_TEST),
--        path.join(DATA_OUT, 'te.t7')
--    )
--    collectgarbage()
end

main()