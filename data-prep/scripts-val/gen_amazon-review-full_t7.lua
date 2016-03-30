require'pl.path'
local ut = require'util.misc'

-- random seed
local seed = 123
torch.manualSeed(seed)
math.randomseed(seed)

--local dataPath = '/mnt/data/datasets/Text/amazon-review-full' -- deepml
local dataPath = '/home/ps/data/datasets/Text/amazon-review-full' -- local

local dataPathT7In = path.join(dataPath, 'word-t7')
local dataPathT7Out = dataPathT7In

-- # to sample
local r = 0.1
local trsize = 3000*1000
local num_sample = math.floor( r * trsize )

--- ensure output path
ut.ensure_path(dataPathT7Out)

print'==> [sub sampling to make trtr, trval]'
require'data-prep.sampleset'.main{
    -- input
    fn_data = path.join(dataPathT7In, 'tr.t7'),
    num_sample = num_sample,
    -- output
    fn_sample = path.join(dataPathT7Out, 'trval.t7'),
    fn_remain = path.join(dataPathT7Out, 'trtr.t7'),
}
