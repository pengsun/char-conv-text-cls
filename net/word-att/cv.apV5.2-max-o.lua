-- concat, seq conv bias
-- strict bow conv, with strict bias
-- mul const for bow-conv.
-- Initialization: weight gaussian, bias zero.
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
    local mconv = nn.OneHotTemporalConvolution(V, HU, KH, {hasBias=true})
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
        return md
    end

    local function make_controller(cw)

        local md = nn.Sequential()
        -- B, M (,V)
        md:add( ohnn.OneHotTemporalBowStack(cw, indUnknown) )
        -- B, Mp (,V)
        md:add( mcontrol )
        -- B, Mp, HU
        md:add( nn.Unsqueeze(1, 2) )
        -- B, 1, Mp, HU
        md:add( cudnn.SpatialAveragePooling(1,cw, 1,cw, 0,0) )
        -- B, 1, M, HU
        md:add( nn.MulConstant(cw, true) )
        md:add( nn.Squeeze(1, 3) )
        -- B, M, HU
        md:add( nn.TemporalAddBias(HU) )
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
    md:add( nn.JoinTable(3, 3) )
    md:add( cudnn.ReLU(true) )
    -- B, M, 2*HU

    -- B, M, 2*HU
    md:add( nn.Max(2) )
    md:add( nn.Dropout() )
    -- B, 2*HU

    -- B, 2*HU
    md:add( nn.Linear(2*HU, K) )
    -- B, K
    md:add( cudnn.LogSoftMax() )
    -- B, K

    local function reinit_params(md)
        local stdv = opt.paramInitBound or 0.08
        print( ('reinit params: weight gaussian, var = %4.3f; bias zero'):format(stdv) )

        local function reinit_weight(w)
            w:normal(0, stdv) -- weight only
        end
        local function reinit_bias(b)
            b:fill(0)
        end

        local function reinit_onehotTempConv_Bias(m)
            local pp = m:parameters()
            local n = #pp; assert(n >= 2);

            for i = 1, n-1 do
                reinit_weight( pp[i] ) -- weight
            end
            reinit_bias( pp[n] ) -- bias
        end

        local function reinit_onehotTempConv_noBias(m)
            local pp = m:parameters()
            local n = #pp

            for i = 1, n do
                reinit_weight( pp[i] ) -- weight
            end
        end

        local function reinit_linear(m)
            local pp = m:parameters()
            assert(#pp == 2)
            reinit_weight( pp[1] )
            reinit_bias( pp[2] )
        end

        reinit_onehotTempConv_Bias(mconv)
        reinit_onehotTempConv_noBias(mcontrol)

        local mlinear = md:findModules('nn.Linear')
        assert(#mlinear == 1)
        reinit_linear(mlinear[1])
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

