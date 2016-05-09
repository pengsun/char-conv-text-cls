-- conv bias, no vocab padding
-- use ohnn (after updating)
require'nn'
require'cudnn'
require'ohnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 300 -- vocabulary/embedding size
    local HU = opt.HU or 190

    local kH = opt.KH or error('no opt.KH')
    local indUnknown = 1 -- for UNKNOWN token
    local mconv = ohnn.OneHotTemporalSeqConvolution(V, HU, kH,
        {hasBias = true, vocabIndPad = 0})

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( mconv )
    md:add( cudnn.ReLU(true) )
    -- B, M-kH+1, HU
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M-kH+1, HU
    md:add( nn.Max(3) )
    md:add( nn.Dropout(0.5, false, true) )
    -- B, 1, HU
    md:add( nn.Reshape(HU, true) )
    -- B, HU

    -- B, HU
    md:add( nn.Linear(HU, K) )
    -- B, K
    md:add(cudnn.LogSoftMax())
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print( ('reinit params gaussian, var = %4.3f'):format(b) )

        local params, _ = md:getParameters()
        params:normal(0, b)
    end
    reinit_params(md)

    mconv:zeroVocabIndPadWeight()

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

