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

    local kH1, kH2 = 2, 3
    local mo = 5

    local function make_cvmax(kH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, mo*HU, kH) )
        -- B, M-kH+1, mo*HU
        md:add( nn.Reshape(M-kH+1, mo*HU, 1, true) )
        -- B, M-kH+1, mo*HU, 1
        md:add( nn.SpatialMaxPooling(1, mo) )
        md:add( nn.Dropout() )
        -- B, M-kH+1, HU, 1
        md:add( nn.Max(2) )
        -- B, HU, 1
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    ct:add( make_cvmax(kH1) )
    ct:add( make_cvmax(kH2) )

    -- B, M (,V)
    md:add( ct )
    -- {B, HU, 1}, {B, HU, 1}
    md:add( nn.JoinTable(2, 3) )
    -- B, 2*HU, 1
    --md:add( nn.ReLU(true) )
    --md:add( nn.Dropout() )
    md:add( nn.Reshape(2*HU, true) )
    -- B, 2*HU

    -- B, 2*HU
    md:add( nn.Linear(2*HU, K) )
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
        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsisten seq length")
    end

    return md, md_reset
end

return this

