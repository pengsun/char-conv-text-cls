require'nn'
require'cudnn'
require'ohnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local V = opt.V or 300 -- vocabulary/embedding size

    local HU = opt.HU or error('no opt.HU')
    assert(type(HU)=='table' and #HU == 3) -- 3 layers

    local kH = opt.KH or error('no opt.KH')
    local pad = 1
    local indUnknown = 1

    local md = nn.Sequential()

    -- B, M (,V)
    md:add( ohnn.OneHotTemporalSeqConvolution(V, HU[1], kH, {
        hasBias = true, padBegLen = 0, padEndLen = 0, padIndValue = indUnknown
    }) )
    md:add( cudnn.ReLU(true) )
    -- B, M, HU1
    md:add( nn.Transpose({2,3}) )
    -- B, HU1, M
    md:add( nn.Unsqueeze(3, 2) )
    -- B, HU1, M, 1

    -- B, HU1, M, 1
    md:add( cudnn.SpatialConvolution(HU[1], HU[2], 1, kH, 1,1, 0,pad) )
    md:add( cudnn.ReLU(true) )
    -- B, HU2, M, 1

    -- B, HU2, M, 1
    md:add( cudnn.SpatialConvolution(HU[2], HU[3], 1, kH, 1,1, 0,pad) )
    md:add( cudnn.ReLU(true) )
    -- B, HU3, M, 1
    md:add( nn.Max(3) )
    md:add( nn.Dropout(0.5, false, true) )
    -- B, HU3, 1

    -- B, HU3, 1
    md:add( nn.Squeeze(2, 2) )
    -- B, HU3
    md:add( nn.Linear(HU[3], K) )
    -- B, K
    md:add(cudnn.LogSoftMax())
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print( ('reinit params gaussian, var = %4.3f'):format(b) )

        local params, _ = md:getParameters()
        params:normal(0, b)
    end
    reinit_params(md)

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

