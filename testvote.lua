require'optim'
require'cudnn'
require'OneHot'

local this = {}

this.main = function (opt)

    local outputs = {}
    local targets
    for i, curOpt in ipairs(opt) do
        print('evaluating model ' .. i)
        print('---------------------')

        outputs[i], targets = dofile('test.lua').main(curOpt) -- OK to overwrite targets
        print(outputs[i]:size())
        print(targets:size())
        print("\n")
    end

    print('perform voting')
    print('--------------')
    local outputsvote = outputs[1]:clone()
    for j = 2, table.getn(outputs) do
        outputsvote:add(outputs[j])
    end

    local conf = optim.ConfusionMatrix( outputsvote:size(2) )
    for n = 1, outputsvote:size(1) do
        conf:add(outputsvote[n], targets[n])
    end
    conf:updateValids()
    print('accuracy = ' .. conf.totalValid)

    return outputsvote, targets
end

return this