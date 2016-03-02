require'nn'
require'cudnn'
require'fbcunn'
require'onehot-temp-conv'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 300 -- vocabulary/embedding size
    local M = opt.seqLength or 291
    local HU = opt.HU or 190

    local kH = 3
    local kpool = 3

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
    md:add( nn.ReLU(true) )
    -- B, M-kH+1, HU
    md:add( nn.TemporalKMaxPooling(kpool) )
    md:add( nn.Dropout() )
    -- B, kpool, HU
    md:add( nn.Reshape(kpool*HU, true) )
    -- B, kpool*HU

    -- B, kpool*HU
    md:add( nn.Linear(kpool*HU, K) )
    -- B, K
    md:add(cudnn.LogSoftMax())
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print( ('reinit params uniform, [%4.3f, %4.3f]'):format(-b,b) )

        local params, _ = md:getParameters()
        params:uniform(-b,b)
    end
    reinit_params(md)

    local function md_reset(md, arg)
        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsisten seq length")
    end

    return md, md_reset
end

return this

