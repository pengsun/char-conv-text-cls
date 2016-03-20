require'nn'
require'cudnn'
require'onehot-temp-conv'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 300 -- vocabulary/embedding size
    local M = opt.seqLength or 291

    local HU = opt.HU or error('no opt.HU')
    assert(type(HU)=='table' and #HU == 4) -- 3 layers + 1 fc layer

    local kH = opt.KH or error('no opt.KH')
    local pool = 2

    local md = nn.Sequential()

    -- B, M (,V)
    md:add( nn.OneHotTemporalConvolution(V, HU[1], kH) )
    md:add( cudnn.ReLU(true) )
    -- B, M-kH+1, HU1
    md:add( nn.Transpose({2,3}) )
    -- B, HU1, M-kH+1
    md:add( nn.Unsqueeze(3, 2) )
    -- B, HU1, M-kH+1, 1
    md:add( cudnn.SpatialMaxPooling(1, pool) )
    md:add( nn.Dropout() )
    -- B, HU1, M', 1

    -- B, HU1, M', 1
    md:add( cudnn.SpatialConvolution(HU[1], HU[2], 1, kH) )
    md:add( cudnn.ReLU(true) )
    -- B, HU2, M'-kH+1, 1
    md:add( cudnn.SpatialMaxPooling(1, pool) )
    md:add( nn.Dropout() )
    -- B, HU2, M'', 1

    -- B, HU2, M'', 1
    md:add( cudnn.SpatialConvolution(HU[2], HU[3], 1, kH) )
    md:add( cudnn.ReLU(true) )
    -- B, HU3, M''-kH+1, 1
    md:add( nn.Max(3) )
    md:add( nn.Dropout() )
    -- B, HU3, 1

    -- B, HU3, 1
    md:add( nn.Squeeze(2, 2) )
    -- B, HU3
    md:add( nn.Linear(HU[3], HU[4]) )
    md:add( cudnn.ReLU(true) )
    md:add( nn.Dropout() )
    -- B, HU4

    -- B, HU4
    md:add( nn.Linear(HU[4], K) )
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
        -- fine to do nothing
    end

    return md, md_reset
end

return this

