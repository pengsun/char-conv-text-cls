require'nn'
require'cudnn'
require'onehot-temp-conv'
require'ConstMul'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 300 -- vocabulary/embedding size
    local M = opt.seqLength or 291
    local HU = opt.HU or 190
    local CW = opt.CW or error('no opt.CW')
    local kH = opt.KH or error('no opt.KH')

    local function get_pad(CW)
        assert(CW %2 == 1)
        return (CW -1)/2
    end
    local pad = get_pad(CW)
    local stride = 1

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
    md:add( nn.ReLU(true) )
    -- B, M-kH+1, HU
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M-kH+1, HU
    md:add( cudnn.SpatialAveragePooling(1,CW, 1,stride, 0,pad) )
    -- B, 1, M-kH+1, HU
    md:add( cudnn.SpatialMaxPooling(1, M-kH+1) )
    -- B, 1, 1, HU
    md:add( nn.Dropout() )
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

    local function md_reset(md, arg)
        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsisten seq length")
    end

    return md, md_reset
end

return this

