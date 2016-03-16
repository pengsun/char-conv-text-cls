require'pl.path'
local ut = require'util.misc'

local dataPath = '/data/datasets/Text/elec' -- deepml
--local dataPath = '/home/ps/data/datasets/Text/elec' -- local

local dataPathWordT7 = path.join(dataPath, 'tr25k-word-t7')
local dataPathTfidf = path.join(dataPath, 'tr25k-tfidf-t7')

--- ensure output path
ut.ensure_path(dataPathTfidf)

print'==> [creating idf: tr.t7 & vocab.t7 to idf.t7]'
require'data-prep.wordtensor-vocab2idf'.main{
    -- input
    fn_data = path.join(dataPathWordT7, 'tr.t7'),
    fn_vocab = path.join(dataPathWordT7, 'vocab.t7'),
    -- output
    fn_idf = path.join(dataPathTfidf, 'idf.t7'),
}

print'==> [creating tf-idf: t7 & vocab.t7 & idf.t7  to tfidf.t7]'
require'data-prep.wordtensor-vocab-idf2tfidf'.main{ -- train
    -- input
    fn_data = path.join(dataPathWordT7, 'tr.t7'),
    fn_vocab = path.join(dataPathWordT7, 'vocab.t7'),
    fn_idf = path.join(dataPathTfidf, 'idf.t7'),
    -- output
    fn_tfidf = path.join(dataPathTfidf, 'tr.t7'),
}
require'data-prep.wordtensor-vocab-idf2tfidf'.main{ -- test
    -- input
    fn_data = path.join(dataPathWordT7, 'te.t7'),
    fn_vocab = path.join(dataPathWordT7, 'vocab.t7'),
    fn_idf = path.join(dataPathTfidf, 'idf.t7'),
    -- output
    fn_tfidf = path.join(dataPathTfidf, 'te.t7'),
}