require'nn'
require'cudnn'
require'fbcunn' -- nn.TemporalKMaxPooling
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

    local kH = 2
    local pool = 2
    local kpool = 10
    local kpoolRatio = 0.5
    local Md = math.max(
        kpool,
        math.floor( (M-kH+1)*kpoolRatio )
    )
    local Mdd = math.floor( (Md-kH+1)/pool )

    local md = nn.Sequential()

    -- B, M (,V)
    md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
    md:add( nn.ReLU(true) )
    -- B, M-kH+1, HU
    md:add( nn.TemporalKMaxPooling(kpool, kpoolRatio) ) -- kpool or half the seq length
    md:add( nn.Dropout() )
    -- B, M', HU

    -- B, M', HU
    md:add( nn.Transpose({2,3}) )
    -- B, HU, M'
    md:add( nn.Reshape(HU, Md, 1) )
    -- B, HU, M', 1
    md:add( cudnn.SpatialConvolution(HU, HU, 1, kH) )
    md:add( cudnn.ReLU(true) )
    -- B, HU, M'-kH+1, 1
    md:add( nn.Max(3) )
    -- B, HU, 1
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

