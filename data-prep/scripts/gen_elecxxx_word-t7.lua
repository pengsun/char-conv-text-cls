require'pl.path'
local ut = require'util.misc'

-- common opt
local numClasses = 2
local vocab_truncate_size = 30000 -- vocabulary control
local trSizeStr = '100k'

local dataPath = '/data/datasets/Text/elec' -- deepml
--local dataPath = '/home/ps/data/datasets/Text/elec' -- local

local dataPathTokCat = path.join(dataPath, '')
local dataPathWordT7 = path.join(dataPath, 'tr'..trSizeStr..'-word-t7')

--- ensure output path
ut.ensure_path(dataPathWordT7)

print'==> [extract vocab: .txt.tok to .vocab]'
require'data-prep.extract_vocab'.main{
    -- input
    fn_tokens = path.join(dataPathTokCat, 'elec-'..trSizeStr..'-train.txt.tok'),
    vocab_truncate_size = vocab_truncate_size,
    -- output
    fn_vocab_freq = path.join(dataPathTokCat,
        'elec-'..trSizeStr..'-train.vocab'
    ),
    fn_vocab_freq_truncate = path.join(dataPathTokCat,
        'elec-'..trSizeStr..'-train-'..vocab_truncate_size..'.vocab'
    ),
}

print'==> [converting to tensors: .txt.tok & .cat to .t7]'
require'data-prep.tokcat2wordtensor'.main{
    -- input
    data_path = dataPathTokCat,
    fn_vocab_freq = 'elec-'..trSizeStr..'-train-'..vocab_truncate_size..'.vocab',
    fn_tok_train = 'elec-'..trSizeStr..'-train.txt.tok',
    fn_cat_train = 'elec-'..trSizeStr..'-train.cat',
    fn_tok_test = 'elec-test.txt.tok',
    fn_cat_test = 'elec-test.cat',
    vocab_cat = {['2']=2, ['1']=1},
    -- output
    data_out = dataPathWordT7,
}
