require'nn'
require'cudnn'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or error('no opt.V') -- char vocabulary size
    local M = opt.seqLength or error('no opt.seqLength')
    local Q = opt.wordLength or error('no opt.wordLength')
    local HU = opt.HU or error('no opt.HU')

    -- 2d filter bank sizes
    local ker = { {1,2},{1,3}, {2,2},{2,3}, {3,2},{3,3} }
    local nBank = #ker

    local function make_cv2dmax(kW, kH)
        local md = nn.Sequential()
        -- B, V, M, Q
        md:add( cudnn.SpatialConvolution(V, HU, kW, kH) )
        -- B, HU, M-kH+1, Q-kW+1
        md:add( cudnn.SpatialMaxPooling(Q-kW+1, M-kH+1) )
        -- B, HU, 1, 1
        return md
    end

    local md = nn.Sequential()

    local convmaxbank = nn.ConcatTable()
    for _, kk in ipairs(ker) do
        convmaxbank:add( make_cv2dmax(unpack(kk)) )
    end


    -- B, M, Q
    md:add( OneHot(V) )
    -- B, M, Q, V
    md:add( nn.Transpose({3,4}, {2,3}) )
    -- B, V, M, Q

    -- B, V, M, Q
    md:add(convmaxbank)
    -- {B,HU,1,1}, ... {B,HU,1,1}
    md:add( nn.JoinTable(2, 4) )
    -- B, nBank*HU, 1, 1
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
    md:add( nn.Reshape(nBank*HU, true) )
    -- B, nBank*HU

    -- B, nBank*HU
    md:add( nn.Linear(nBank*HU, K) )
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

