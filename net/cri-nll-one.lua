require'nn'
require'cudnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    print('creating NLL Criterion x1')
    return nn.ClassNLLCriterion()
end

return this

