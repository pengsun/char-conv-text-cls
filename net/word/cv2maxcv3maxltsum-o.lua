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

    local kH1, kH2 = 2, 3

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

    local function make_bow()
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.LookupTable(V, HUBOW) )
        -- B, M, HUBOW
        md:add( nn.Sum(2) )
        -- B, HUBOW
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    ct:add( make_cvmax(kH1) )
    ct:add( make_cvmax(kH2) )
    ct:add( make_bow() )

    -- B, M (,V)
    md:add( ct )
    -- {B, HU}, {B, HU}, {B, HUBOW}
    md:add( nn.JoinTable(2, 2) )
    -- B, 2*HU+HUBOW
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
    -- B, 2*HU+HUBOW

    -- B, 2*HU+HUBOW
    md:add( nn.Linear(2*HU+HUBOW, K) )
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

