require'pl.path'
local ut = require'util.misc'

--- common opt
local numClasses = 14

local dataPath = '/mnt/data/datasets/Text/dbpedia' -- deepml
--local dataPath = '/home/ps/data/dbpedia' -- local

local dataPathTokCat = path.join(dataPath, 'tok-cat')
local dataPathOut = path.join(dataPath, 'char2d-t7')

-- ensure output path
ut.ensure_path(dataPathOut)


print'==> [converting to tensors: .txt.tok & .cat to .t7]'
local get_vocab_cat = function ()
    local cat = {}
    for i = 1, numClasses do cat[tostring(i)] = i end
    return cat
end
require'data-prep.tokcat2char2dtensor'.main{
    -- input
    data_path = dataPathTokCat,
    fn_vocab_freq = 'train-30000.vocab',
    fn_tok_train = 'train.txt.tok',
    fn_cat_train = 'train.cat',
    fn_tok_test = 'test.txt.tok',
    fn_cat_test = 'test.cat',
    vocab_cat = get_vocab_cat(),
    -- output
    data_out = dataPathOut,
}

