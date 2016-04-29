-- conv bias, zero padding
-- padding for sequence
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
    local HU = opt.HU or 190

    local kH = opt.KH or error('no opt.KH')
    local indUnknown = 1 -- for UNKNOWN token
    local mconv = nn.OneHotTemporalConvolution(V, HU, kH, {hasBias = true})

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
    md:add( cudnn.ReLU(true) )
    -- B, M, HU
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M, HU
    md:add( nn.Max(3) )
    md:add( nn.Dropout() )
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

    print('setting unknown index ' .. indUnknown)
    mconv:setPadding(indUnknown):zeroPaddingWeight()

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

