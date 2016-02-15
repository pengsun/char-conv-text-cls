require'nn'
require'cudnn'
require'onehot-temp-conv'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or error('no opt.V') -- char vocabulary size
    local M = opt.seqLength or error('no opt.seqLength')
    local Q = opt.wordLength or error('no opt.wordLength')
    local HU = opt.HU or error('no opt.HU')

    local kH1, kH2 = 2, 3

    local function make_cvmax(kH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
        -- B, M-kH1+1, HU
        md:add( nn.TemporalMaxPooling(M-kH+1) )
        -- B, 1, HU
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    ct:add( make_cvmax(kH1) )
    ct:add( make_cvmax(kH2) )

    -- B, M, Q
    md:add( OneHot(V) )
    -- B, M, Q, V
    md:add( nn.Transpose({3,4}, {2,3}) )
    -- B, V, M, Q
    md:add( ct )
    -- {B, 1, HU}, {B, 1, HU}
    md:add( nn.JoinTable(3, 3) )
    -- B, 1, 2*HU
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
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

