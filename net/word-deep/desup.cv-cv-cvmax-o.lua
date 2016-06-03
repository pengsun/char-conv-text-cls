require'nn'
require'cudnn'
require'ohnn'
require'nngraph'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local V = opt.V or 300 -- vocabulary/embedding size

    local HU = opt.HU or error('no opt.HU')
    assert(type(HU)=='table' and #HU == 3) -- 3 layers

    local kH = opt.KH or error('no opt.KH')

    local function get_pad(width)
        assert(width %2 == 1)
        return (width -1)/2
    end
    local pad = get_pad(kH)

    local indUnknown = 1

    local function make_stackedConvLayers()

        local function make_seqConvLayer(nodeX, Hin, Hout)
            local mconv = ohnn.OneHotTemporalSeqConvolution(Hin, Hout, kH, {
                hasBias = true, padBegLen = pad, padEndLen = pad, padIndValue = indUnknown
            })

            -- B, M (,V)
            local nodeY = mconv (nodeX)
            -- B, M, H
            local nodeY2 = nn.Transpose({2,3}) (nodeY)
            -- B, H, M
            local nodeY3 = nn.Unsqueeze(3, 2) (nodeY2)
            -- B, H, M, 1
            local nodeY4 = cudnn.ReLU(true) (nodeY3)
            -- B, H, M, 1

            return nodeY4
        end

        local function make_spatialConvLayer(nodeX, Hin, Hout)
            local mconv = cudnn.SpatialConvolution(Hin, Hout, 1, kH, 1,1, 0,pad)

            -- B, HIn, M, 1
            local nodeY = mconv (nodeX)
            -- B, Hout, M, 1
            local nodeY2 = cudnn.ReLU(true) (nodeY)
            -- B, Hout, M, 1

            return nodeY2
        end

        local x = nn.Identity() ()
        local x1 = make_seqConvLayer(x, V, HU[1])
        local x2 = make_spatialConvLayer(x1, HU[1], HU[2])
        local x3 = make_spatialConvLayer(x2, HU[2], HU[3])

        -- Input: B, M (,V)
        local m = nn.gModule({x}, {x1, x2, x3})
        -- Output: {B, HU1, M, 1}, {B, HU2, M, 1}, {B, HU3, M, 1}
        return m
    end

    local function make_parOutputLayers()
        local function make_outputLayer(nHidUnits)
            local md = nn.Sequential()
            -- B, H, M, 1
            md:add( nn.Max(3) )
            -- B, H, 1
            md:add( nn.Squeeze(3) )
            -- B, H
            md:add( nn.Linear(nHidUnits, K) )
            -- B, K
            return md
        end

        local pt = nn.ParallelTable()
        -- {B, HU1, M, 1}, {B, HU2, M, 1}, {B, HU3, M, 1}
        pt:add( make_outputLayer(HU[1]) )
        pt:add( make_outputLayer(HU[2]) )
        pt:add( make_outputLayer(HU[3]) )
        -- {B, K}, {B, K}, {B, K}
        return pt
    end

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( make_stackedConvLayers() )
    -- {B, HU1, M, 1}, {B, HU2, M, 1}, {B, HU3, M, 1}
    md:add( make_parOutputLayers() )
    -- {B, K}, {B, K}, {B, K}
    md:add( nn.CAddTable(true) )
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

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this

