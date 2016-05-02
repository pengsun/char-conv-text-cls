-- type II, seq conv bias
-- strict bow conv
-- Initialization: gaussian for all
-- seq-conv strict align by padding input.
require'nn'
require'cudnn'
require'onehot-temp-conv'
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
    local mconv = nn.OneHotTemporalConvolution(V, HU, KH, {hasBias = true})
    local mcontrol = nn.LookupTable(V, HU)

    local function make_cv(kH)
        -- pad for input
        local function get_pad()
            assert(kH %2 == 1)
            return (kH -1)/2
        end
        local pad = get_pad()
        local inputDim = 1
        local mPadLeft = nn.Padding(1, -pad, inputDim, indUnknown)
        local mPadRight = nn.Padding(1, pad, inputDim, indUnknown)
        -- turn off gradInput for padding module
        local function null_updateGradInput(self, input, gradOutput)
            self.gradInput = torch.Tensor():typeAs(gradOutput)
            return self.gradInput
        end
        mPadLeft.updateGradInput = null_updateGradInput
        mPadRight.updateGradInput = null_updateGradInput

        local md = nn.Sequential()
        -- B, M (,V)
        md:add( mPadLeft )
        md:add( mPadRight )
        -- B, M+kH-1
        md:add( mconv )
        -- B, M, HU
        md:add( cudnn.ReLU(true) )
        -- B, M, HU
        return md
    end

    local function make_controller(p)

        local md = nn.Sequential()
        -- B, M (,V)
        md:add( ohnn.OneHotTemporalBowStack(p, indUnknown) )
        -- B, Mp (,V)
        md:add( mcontrol )
        -- B, Mp, HU
        md:add( nn.Unsqueeze(1, 2) )
        -- B, 1, Mp, HU
        md:add( cudnn.SpatialAveragePooling(1,p, 1,p, 0,0) )
        -- B, 1, M, HU
        md:add( nn.MulConstant(p, true) )
        md:add( nn.Squeeze(1, 3) )
        -- B, M, HU
        md:add( nn.TemporalAddBias(HU) )
        md:add( cudnn.Sigmoid(true) )
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

    print('setting unknown token index')
    mconv:setPadding(indUnknown):zeroPaddingWeight()
    mcontrol:setPadding(indUnknown)
    mcontrol.weight:select(1, indUnknown):fill(0)

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

