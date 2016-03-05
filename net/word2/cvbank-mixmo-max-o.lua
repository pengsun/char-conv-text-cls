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

    local mo = opt.MO or error('no opt.MO')

    local function check_kernels()
        local khkh = opt.KHKH or error('no opt.KHKH')
        local padpad = opt.PADPAD or error('no opt.PADPAD')

        assert(type(khkh)=='table' and type(padpad) == 'table')
        assert(#khkh == #padpad)
        return khkh, padpad
    end
    local khkh, padpad = check_kernels()
    local nb = #khkh

    local function check_Md()
        local Md = M - khkh[1] + 1 + padpad[1]
        for i = 2, #khkh do
            local Md_ = M - khkh[i] + 1 + padpad[i]
            assert(Md == Md_,
                ("cvbank %d: kernel size %d and padding %d mismatch"):format(i, khkh[i], padpad[i])
            )
        end

        return Md
    end
    local Md = check_Md()

    local function make_cv(kH, pad)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, mo*HU, kH) )
        -- B, M-kH+1, mo*HU
        md:add( nn.Reshape(1, M-kH+1, mo*HU, true) )
        -- B, 1, M-kH+1, mo*HU
        md:add( nn.SpatialReplicationPadding(0,0,0,pad) )
        -- B, 1, M', mo*HU
        md:add( nn.Reshape(Md, mo*HU, 1, true) )
        -- B, M', mo*HU, 1
        return md
    end

    local md = nn.Sequential()

    local ct = nn.ConcatTable()
    for i, kH in ipairs(khkh) do
        local pad = padpad[i]
        ct:add( make_cv(kH, pad) )
    end

    -- B, M (,V)
    md:add( ct )
    -- {B, M', mo*HU, 1}, ..., {B, M', mo*HU, 1}
    md:add( nn.JoinTable(4, 4) )
    md:add( cudnn.ReLU(true) )
    -- B, M', mo*HU, nb

    -- B, M', mo*HU, nb
    md:add( nn.Reshape(Md, nb*HU, mo) )
    -- B, M', nb*HU, mo
    md:add( cudnn.SpatialMaxPooling(mo, 1) )
    -- B, M', nb*HU, 1
    md:add( nn.Dropout() )
    md:add( nn.Max(2))
    -- B, nb*HU, 1
    md:add( nn.Reshape(nb*HU, true) )
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

