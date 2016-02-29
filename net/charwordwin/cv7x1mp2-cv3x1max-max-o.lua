require'nn'
require'cudnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or error('no opt.V') -- char vocabulary size
    local M = opt.seqLength or error('no opt.seqLength')
    local Q = opt.winSize or error('no opt.winSize')
    local HU = opt.HU or error('no opt.HU')

    local kW1, kW2 = 7, 3
    local pool = 2

    local md = nn.Sequential()

    -- B, M, Q
    md:add( OneHot(V) )
    -- B, M, Q, V
    md:add( nn.Transpose({3,4}, {2,3}) )
    -- B, V, M, Q

    -- B, V, M, Q
    md:add( cudnn.SpatialConvolution(V, HU, kW1, 1) )
    md:add( cudnn.ReLU(true) )
    md:add( cudnn.SpatialMaxPooling(pool, 1) )
    md:add( nn.Dropout() )
    -- B, HU, M, Q'

    -- B, HU, M, Q'
    md:add( cudnn.SpatialConvolution(HU, HU, kW2, 1) )
    md:add( cudnn.ReLU(true) )
    md:add( nn.Max(4) )
    -- B, HU, M
    md:add( nn.Dropout() )
    -- B, HU, M

    -- B, HU, M
    md:add( nn.Max(3) )
    -- B, HU
    md:add( nn.Linear(HU, K) )
    -- B, K
    md:add( cudnn.LogSoftMax() )
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print( ('reinit params uniform, [%4.3f, %4.3f]'):format(-b,b) )

        local params, _ = md:getParameters()
        params:uniform(-b,b)
    end
    reinit_params(md)

    local function md_reset(md, arg)
        --local newM = arg.seqLength or error('no seqLength')
        --assert(newM==M, "inconsisten seq length")
    end

    return md, md_reset
end

return this

