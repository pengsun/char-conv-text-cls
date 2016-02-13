require'pl.path'
local ut = require'util.misc'

--- common opt
local numClasses = 5
local vocab_truncate_size = 30000 -- vocabulary control

local dataPath = '/mnt/data/datasets/Text/amazon-review-full' -- deepml
--local dataPath = '/home/ps/data/amazon-review-full' -- local

local dataPathTokCat = path.join(dataPath, 'tok-cat')
local dataPathWordT7 = path.join(dataPath, 'word-t7')

--- ensure output path
ut.ensure_path(dataPathTokCat)
ut.ensure_path(dataPathWordT7)

---
print'==> [csv to text and category: .csv to .txt & .cat]'
--require('util.data.csv2txtcat').main{ -- train
--    -- input
--    path_csv = dataPath,
--    fn_csv = 'train.csv',
--    -- output
--    path_txt_cat = dataPathTokCat,
--    fn_txt = 'train.txt',
--    fn_cat = 'train.cat',
--}
--require('util.data.csv2txtcat').main{ -- test
--    -- input
--    path_csv = dataPath,
--    fn_csv = 'test.csv',
--    -- output
--    path_txt_cat = dataPathTokCat,
--    fn_txt = 'test.txt',
--    fn_cat = 'test.cat',
--}

---
print'==> [tokenize: .txt to .txt.tok]'
--require'util.data.txt2tok'.main{ -- train
--    -- input
--    path_data = dataPathTokCat,
--    fn_txt = 'train.txt',
--}
require'util.data.txt2tok'.main{ -- test
    -- input
    path_data = dataPathTokCat,
    fn_txt = 'test.txt',
}

---
print'==> [extract vocab: .txt.tok to .vocab]'
require'util.data.extract_vocab'.main{
    -- input
    fn_tokens = path.join(dataPathTokCat, 'train.txt.tok'),
    vocab_truncate_size = vocab_truncate_size,
    -- output
    fn_vocab_freq = path.join(dataPathTokCat, 'train.vocab'),
    fn_vocab_freq_truncate = path.join(dataPathTokCat, 'train-' .. vocab_truncate_size .. '.vocab'),
}

---
print'==> [converting to tensors: .txt.tok & .cat to .t7]'
local get_cat = function ()
    local cat = {}
    for i = 1, numClasses do cat[tostring(i)] = i end
    return cat
end
require'util.data.tokcat2wordtensor'.main{
    -- input
    data_path = dataPathTokCat,
    fn_vocab_freq = 'train-' .. vocab_truncate_size .. '.vocab',
    fn_tok_train = 'train.txt.tok',
    fn_cat_train = 'train.cat',
    fn_tok_test = 'test.txt.tok',
    fn_cat_test = 'test.cat',
    vocab_cat = get_cat(),
    -- output
    data_out = dataPathWordT7,
}

