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
    local M = opt.seqLength or 291
    local HU = opt.HU or 190
    local KHKH = opt.KHKH or error('no opt.KHKH')
    local CW = opt.CW or error('no opt.CW')

    assert(type(KHKH)=='table')
    local nb = #KHKH

    local function make_cvbank(khkh)
        local function make_cv(kH)
            local md = nn.Sequential()
            -- B, M (,V)
            md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
            -- B, M-kH+1, HU
            md:add( nn.Padding(2, kH-1) )
            -- B, M, HU
            return md
        end

        local ct = nn.ConcatTable()
        for _, kH in ipairs(khkh) do
            ct:add( make_cv(kH) )
        end

        local md = nn.Sequential()
        -- B, M (,V)
        md:add( ct )
        -- {B, M, HU}, ..., {B, M, HU}
        md:add( nn.JoinTable(3, 3) )
        -- B, M, nb*HU
        md:add( cudnn.ReLU(true) )
        -- B, M, nb*HU
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
        md:add( nn.OneHotTemporalConvolution(V, nb*HU, 1) )
        -- B, M, nb*HU
        md:add( nn.Unsqueeze(1, 2) )
        -- B, 1, M, nb*HU
        md:add( cudnn.SpatialAveragePooling(1,cw, 1,stride, 0,pad) )
        -- B, 1, M, nb*HU
        md:add( nn.Transpose({2,3}) )
        -- B, M, 1, nb*HU
        md:add( cudnn.SpatialSoftMax() )
        -- B, M, 1, nb*HU
        md:add( nn.Squeeze(2, 3) )
        -- B, M, nb*HU
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
        :add( make_cvbank(KHKH) )
        :add( make_controller(CW) )

    -- B, M (,V)
    md:add( ct )
    -- {B, M, nb*HU}, {B, M, nb*HU}
    md:add( nn.CMulTable(3, 3) )
    -- B, M, nb*HU

    -- B, M, nb*HU
    md:add( nn.Sum(2, 3, false) ) -- average = false
    md:add( nn.Dropout() )
    -- B, nb*HU

    -- B, nb*HU
    md:add( nn.Linear(nb*HU, K) )
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

