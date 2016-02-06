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
    local HUBOW = 20 -- hidden units for fc layer over bow

    local kH1, kH2, kH3 = 2, 3, 4

    local function make_cvmax(kH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
        -- B, M-kH1+1, HU
        md:add( nn.TemporalMaxPooling(M-kH+1) )
        -- B, 1, HU
        md:add( nn.View(-1):setNumInputDims(2) )
        -- B, HU
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    ct:add( make_cvmax(kH1) )
    ct:add( make_cvmax(kH2) )
    ct:add( make_cvmax(kH3) )

    -- B, M (,V)
    md:add( ct )
    -- {B, HU}, {B, HU}, {B, HU}
    md:add( nn.JoinTable(2, 2) )
    -- B, 3*HU
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
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
        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsisten seq length")
    end

    return md, md_reset
end

return this

