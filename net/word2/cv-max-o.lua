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
    local HU = opt.HU or 190

    local kH = opt.KH or error('no opt.KH')
    local indUnknown = 1 -- for UNKNOWN token
    local mconv = nn.OneHotTemporalConvolution(V, HU, kH)


    local md = nn.Sequential()
    -- B, M (,V)
    md:add( mconv )
    md:add( cudnn.ReLU(true) )
    -- B, M-kH+1, HU
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M-kH+1, HU
    md:add( cudnn.SpatialMaxPooling(1, M-kH+1) )
    md:add( nn.Dropout() )
    -- B, 1, 1, HU
    md:add( nn.Reshape(HU, true) )
    -- B, HU

    -- B, HU
    md:add( nn.Linear(HU, K) )
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

--    print('setting unknown index ' .. indUnknown)
--    mconv:setPadding(indUnknown):zeroPaddingWeight()

    local function md_reset(md, arg)
        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsisten seq length")
    end

    return md, md_reset
end

return this

