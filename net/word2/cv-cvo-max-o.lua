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
    local cvo = opt.CVO or error('no opt.CVO')

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( nn.OneHotTemporalConvolution(V, cvo *HU, kH) )
    md:add( cudnn.ReLU(true) )
    -- B, M-kH+1, cvo*HU
    md:add( nn.Reshape(M-kH+1, cvo *HU, 1, true) )
    -- B, M-kH+1, cvo*HU, 1
    md:add( nn.Transpose({2,3}) )
    -- B, cvo*HU, M-kH+1, 1
    md:add( cudnn.SpatialConvolution(cvo*HU, HU, 1,1) )
    md:add( cudnn.ReLU(true) )
    md:add( nn.Dropout() )
    -- B, HU, M-kH+1, 1
    md:add( nn.Max(3) )
    -- B, HU, 1
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

