require'pl.path'
local ut = require'util.misc'

--- common opt
local numClasses = 2 -- positive/negative
local vocab_truncate_size = 30000 -- vocabulary control

local dataPath = '/data/datasets/Text/amazon-review-polarity' -- deepml
--local dataPath = '/home/ps/data/amazon-review-polarity' -- local

local dataPathTokCat = path.join(dataPath, 'tok-cat-rie')
local dataPathWordT7 = path.join(dataPath, 'word-t7-rie')

--- ensure output path
ut.ensure_path(dataPathTokCat)
ut.ensure_path(dataPathWordT7)


print'==> [csv to text and category: .csv to .txt & .cat]'
require('data-prep.csv2txtcat-rie').main{ -- train
    -- input
    pl_script = 'data-prep/convert-multifields.pl',
    input_path = path.join(dataPath, 'train.csv'),
    -- output
    output_path_name = path.join(dataPathTokCat, 'train'),
}
require('data-prep.csv2txtcat-rie').main{ -- train
    -- input
    pl_script = 'data-prep/convert-multifields.pl',
    input_path = path.join(dataPath, 'test.csv'),
    -- output
    output_path_name = path.join(dataPathTokCat, 'test'),
}


print'==> [tokenize: .txt to .txt.tok]'
require'data-prep.txt2tok'.main{ -- train
    -- input
    path_data = dataPathTokCat,
    fn_txt = 'train.txt',
    pl_script = 'data-prep/to_tokens-url.pl'
}
require'data-prep.txt2tok'.main{ -- test
    -- input
    path_data = dataPathTokCat,
    fn_txt = 'test.txt',
    pl_script = 'data-prep/to_tokens-url.pl'
}


print'==> [extract vocab: .txt.tok to .vocab]'
require'data-prep.extract_vocab'.main{
    -- input
    fn_tokens = path.join(dataPathTokCat, 'train.txt.tok'),
    vocab_truncate_size = vocab_truncate_size,
    -- output
    fn_vocab_freq = path.join(dataPathTokCat, 'train.vocab'),
    fn_vocab_freq_truncate = path.join(dataPathTokCat, 'train-' .. vocab_truncate_size .. '.vocab'),
}


print'==> [converting to tensors: .txt.tok & .cat to .t7]'
local get_cat = function ()
    local cat = {}
    for i = 1, numClasses do cat[tostring(i)] = i end
    return cat
end
require'data-prep.tokcat2wordtensor'.main{
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

