--- one-hot word convnet + captial feature, full connection
require'cudnn'
require'onehot-temp-conv'

local this = {}

this.main = function(opt)
    -- option: common
    local opt = opt or {}
    local K = opt.numClasses or error('no numClasses') -- #classes
    local B = opt.batSize or error('no batSize') -- batch size
    local V = opt.V or error('no V')
    local C = opt.C or error('no C')
    local HU = opt.HU or error('no HU')

    local kH = 3

    local function create_md_one()
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
        -- B, M', HU
        return md
    end

    local function create_md_two(opt)
        local kW = 5

        local md = nn.Sequential()
        -- B, M, Q (,C)
        md:add( OneHot(C) )
        -- B, M, Q, C
        md:add( nn.Transpose({3,4}, {2,3}) )
        -- B, C, M, Q

        -- B, C, M, Q
        md:add( cudnn.SpatialConvolution(C, HU, kW, kH) )
        -- B, HU, M', Q'
        md:add( nn.Max(4) )
        -- B, HU, M'
        md:add( nn.Transpose({2,3}) )
        -- B, M', HU

        return md
    end

    -- the sub nets
    local parNet = nn.ParallelTable()
    parNet:add( create_md_one() )
    parNet:add( create_md_two() )

    -- net
    local md = nn.Sequential()

    -- {B, M (,V)}, {B, M, Q (,C)}
    md:add(parNet)
    -- {B, M', HU}, {B, M', HU}
    md:add(nn.CAddTable())
    -- B, M', HU
    md:add(cudnn.ReLU(true))
    md:add(nn.Max(2))
    -- B, HU
    md:add(nn.Dropout())
    -- B, HU

    -- B, HU
    md:add(nn.Linear(HU, K))
    -- B, K
    md:add(cudnn.LogSoftMax())
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print('reinit params uniform,  [-' .. b .. ', ' .. b .. ']')

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