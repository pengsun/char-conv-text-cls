-- type II, bias
require'nn'
require'cudnn'
require'onehot-temp-conv'

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
    local mconv = nn.OneHotTemporalConvolution(V, HU, KH, {hasBias = true})
    local mcontrol = nn.OneHotTemporalConvolution(V, HU, 1, {hasBias = true})

    local function make_cv(kH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( mconv )
        -- B, M-kH+1, HU
        md:add( nn.Padding(2, kH-1) )
        -- B, M, HU
        md:add( cudnn.ReLU(true) )
        -- B, M, HU
        return md
    end

    local function make_controller(cw)

        local function get_pad()
            assert(cw %2 == 1)
            return (cw -1)/2
        end
        local pad = get_pad(cw)
        local stride = 1

        local md = nn.Sequential()
        -- B, M (,V)
        md:add( mcontrol )
        -- B, M, HU
        md:add( nn.Unsqueeze(1, 2) )
        -- B, 1, M, HU
        md:add( cudnn.SpatialAveragePooling(1,cw, 1,stride, 0,pad) )
        md:add( nn.MulConstant(cw, true) )
        md:add( cudnn.Sigmoid() )
        -- B, 1, M, HU
        md:add( nn.Squeeze(1, 3) )
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
    md:add( nn.CMulTable(3, 3) )
    -- B, M, HU

    -- B, M, HU
    md:add( nn.Max(2) )
    md:add( nn.Dropout() )
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

    print('setting unknown token index')
    mconv:setPadding(indUnknown):zeroPaddingWeight()
    mcontrol:setPadding(indUnknown):zeroPaddingWeight()

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

