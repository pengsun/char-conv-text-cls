-- type I, weight sharing, softmax,
-- use ohnn (after updating for padding support) for both seq-conv and bow-conv
require'nn'
require'cudnn'
require'ohnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local V = opt.V or 300 -- vocabulary/embedding size
    local HU = opt.HU or 190
    local KH = opt.KH or error('no opt.kH')
    local CW = opt.CW or error('no opt.CW')

    local indUnknown = 1

    -- pad for input
    local function get_pad(width)
        assert(width %2 == 1)
        return (width -1)/2
    end
    local padSeq = get_pad(KH)
    local padSow = get_pad(CW)

    -- seq-conv and bow-conv
    local mconv = ohnn.OneHotTemporalSeqConvolution(V, HU, KH,
        {hasBias = true, padBegLen = padSeq, padEndLen = padSeq, padIndValue = indUnknown})
    local mcontrol = ohnn.OneHotTemporalSowConvolution(V, 1, CW,
        {hasBias = true, padBegLen = padSow, padEndLen = padSow})

    local function make_cv(kH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( mconv )
        -- B, M, HU
        md:add( cudnn.ReLU(true) )
        -- B, M, HU
        return md
    end

    local function make_controller(cw)

        local md = nn.Sequential()
        -- B, M (,V)
        md:add( mcontrol )
        -- B, M, 1
        md:add( nn.Unsqueeze(1,2) )
        -- B, 1, M, 1
        md:add( cudnn.SpatialSoftMax() )
        md:add( nn.Squeeze(1, 3) )
        -- B, M, 1
        md:add( nn.Squeeze(2, 2) )
        -- B, M
        md:add( nn.Replicate(HU, 3) )
        -- B, M, HU
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
        :add( make_cv(KH) )
        :add( make_controller(CW) )

    -- B, M (,V)
    md:add( ct )
    -- {B, M, HU}, {B, M, HU}
    md:add( nn.CMulTable() )
    -- B, M, HU

    -- B, M, HU
    md:add( nn.Max(2) )
    md:add( nn.Dropout(0.5, false, true) )
    -- B, HU

    -- B, HU
    md:add( nn.Linear(HU, K) )
    -- B, K
    md:add( cudnn.LogSoftMax() )
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print( ('reinit params gaussian, var = %4.3f'):format(b) )

        local params, _ = md:getParameters()
        params:normal(0, b)
    end
    reinit_params(md)

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

