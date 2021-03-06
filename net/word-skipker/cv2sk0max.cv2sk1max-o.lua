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
    local HU = opt.HU or error('no opt.HU')

    local kH1, kH2 = 2, 2
    local kS1, kS2 = {0}, {1}
    local H1, H2 = 300, 300
    assert(HU == (H1+H2))

    local function make_cvmax(kH, kS, HUHU)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolutionSkipKer(V, HUHU, kH, kS) )
        -- B, M-kHH+1, HU
        md:add( nn.Max(2) )
        -- B, HU
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    ct:add( make_cvmax(kH1, kS1, H1) )
    ct:add( make_cvmax(kH2, kS2, H2) )

    -- B, M (,V)
    md:add( ct )
    -- {B, HU1}, {B, HU2}, {B, HU3}
    md:add( nn.JoinTable(2, 2) )
    -- B, HU
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
    md:add( nn.Reshape(HU, true) )
    -- B, HU

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
        -- Okay to do nothing
    end

    return md, md_reset
end

return this

