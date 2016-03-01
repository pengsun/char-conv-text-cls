require'nn'
require'cudnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    print('creating NLL Criterion x2')
    local repeatTarget = true
    return nn.ParallelCriterion(repeatTarget)
            :add(nn.ClassNLLCriterion())
            :add(nn.ClassNLLCriterion())
end

return this