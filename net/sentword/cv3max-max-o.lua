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
    local S = opt.numSent or error('no nop.numSent')
    local M = opt.seqLength or 291
    local HU = opt.HU or 190

    local kH = 3

    local md = nn.Sequential()
    local reshapFPOnly = nn.Reshape(B*S, M, false)
    reshapFPOnly.updateGradInput = function() end -- no gradOutput

    -- B, S, M (,V)
    md:add( reshapFPOnly )
    -- BS, M (,V)
    md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
    md:add( cudnn.ReLU(true) )
    -- BS, M-kH+1, HU
    md:add( nn.Max(2) )
    -- BS, HU
    md:add( nn.Reshape(B, 1, S, HU, false) )
    -- B, 1, S, HU
    md:add( nn.Transpose({2,4}) )
    -- B, HU, S, 1

    -- B, HU, S, 1
    md:add( nn.Max(3) )
    md:add( nn.Dropout() )
    -- B, HU, 1

    -- B, HU, 1
    md:add( nn.Reshape(B, HU, false) )
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
        local newB = arg.batSize or error('no batSize')
        local newS = arg.numSent or error('no numSent')
        local newM = arg.seqLength or error('no seqLength')

        assert(newB==B, "inconsistent batSize")
        assert(newS==S, "inconsistent numSent")
        assert(newM==M, "inconsistent seq length")
    end

    return md, md_reset
end

return this