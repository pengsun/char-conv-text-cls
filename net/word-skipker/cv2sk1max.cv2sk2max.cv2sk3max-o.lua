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

    local kH1, kH2, kH3 = 2, 2, 2
    local kS1, kS2, kS3 = {1}, {2}, {3}

    local function make_cvmax(kH, kS)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolutionSkipKer(V, HU, kH, kS) )
        -- B, M-kHH+1, HU
        md:add( nn.Max(2) )
        -- B, HU
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    ct:add( make_cvmax(kH1, kS1) )
    ct:add( make_cvmax(kH2, kS2) )
    ct:add( make_cvmax(kH3, kS3) )

    -- B, M (,V)
    md:add( ct )
    -- {B, HU}, {B, HU}, {B, HU}
    md:add( nn.JoinTable(2, 2) )
    -- B, 3*HU
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
    md:add( nn.Reshape(3*HU, true) )
    -- B, 3*HU

    -- B, 3*HU
    md:add( nn.Linear(3*HU, K) )
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
        -- Okay to do nothing
    end

    return md, md_reset
end

return this

