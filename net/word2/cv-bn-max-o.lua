-- conv bias, batch normalization, zero padding
require'nn'
require'cudnn'
require'onehot-temp-conv'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local V = opt.V or 300 -- vocabulary/embedding size
    local HU = opt.HU or 190

    local kH = opt.KH or error('no opt.KH')
    local indUnknown = 1 -- for UNKNOWN token
    local mconv = nn.OneHotTemporalConvolution(V, HU, kH, {hasBias = true})

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( mconv )
    -- B, M-kH+1, HU
    md:add( nn.Transpose({2,3}) )
    -- B, HU, M-kH+1
    md:add( nn.Unsqueeze(3, 2) )
    -- B, HU, M-kH+1, 1
    md:add( cudnn.SpatialBatchNormalization(HU) )
    md:add( cudnn.ReLU(true) )
    -- B, HU, M-kH+1, 1
    md:add( nn.Squeeze(3, 3) )
    -- B, HU, M-kH+1
    md:add( nn.Max(3) )
    -- B, HU
    md:add( nn.Dropout() )
    -- B, HU

    -- B, HU
    md:add( nn.Linear(HU, K) )
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

    print('setting unknown index ' .. indUnknown)
    mconv:setPadding(indUnknown):zeroPaddingWeight()

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

