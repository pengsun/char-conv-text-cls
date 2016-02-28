require'pl.path'
local ut = require'util.misc'

--- common opt
local numClasses = 2

--local dataPath = '/mnt/data/datasets/Text/elec' -- deepml
local dataPath = '/home/ps/data/datasets/Text/elec' -- local

local dataPathTokCat = dataPath
local dataPathOut = path.join(dataPath, 'tr25k-char-t7')

-- ensure output path
ut.ensure_path(dataPathOut)


print'==> [converting to tensors: .txt.tok & .cat to .t7]'
local get_vocab_cat = function ()
    local cat = {}
    for i = 1, numClasses do cat[tostring(i)] = i end
    return cat
end
require'data-prep.tokcat2chartensor'.main{
    -- input
    data_path = dataPathTokCat,
    fn_tok_train = 'elec-25k-train.txt.tok',
    fn_cat_train = 'elec-25k-train.cat',
    fn_tok_test = 'elec-test.txt.tok',
    fn_cat_test = 'elec-test.cat',
    vocab_cat = get_vocab_cat(),
    -- output
    data_out = dataPathOut,
}

