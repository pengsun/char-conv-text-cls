require'pl.path'
local ut = require'util.misc'

--- common opt
local numClasses = 2

--local dataPath = '/mnt/data/datasets/Text/imdb' -- deepml
local dataPath = '/data/datasets/Text/imdb' -- deepml2
--local dataPath = '/home/ps/data/datasets/Text/imdb' -- local

local dataPathTokCat = dataPath
local dataPathOut = path.join(dataPath, 'char-t7')

-- ensure output path
ut.ensure_path(dataPathOut)


print'==> [converting to tensors: .txt.tok & .cat to .t7]'
require'data-prep.tokcat2chartensor'.main{
    -- input
    data_path = dataPathTokCat,
    fn_tok_train = 'imdb-train.txt.tok',
    fn_cat_train = 'imdb-train.cat',
    fn_tok_test = 'imdb-test.txt.tok',
    fn_cat_test = 'imdb-test.cat',
    vocab_cat = {['pos']=2, ['neg']=1},
    -- output
    data_out = dataPathOut,
}

