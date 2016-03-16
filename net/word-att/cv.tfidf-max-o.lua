--- one-hot word convnet + captial feature, full connection
require'cudnn'
require'onehot-temp-conv'

local this = {}

this.main = function(opt)
    -- option: common
    local opt = opt or {}
    local K = opt.numClasses or error('no numClasses') -- #classes
    local V = opt.V or error('no V')
    local HU = opt.HU or error('no HU')
    local KH = opt.KH or error('no opt.kH')

    local function make_cv(KH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, HU, KH) )
        -- B, M-KH+1, HU
        md:add( nn.Padding(2, KH-1) )
        -- B, M, HU
        md:add( cudnn.ReLU(true) )
        -- B, M, HU
        return md
    end

    local function make_tfidf(opt)
        local md = nn.Sequential()
        -- B, M
        md:add( nn.Identity() )
        -- B, M
        md:add( nn.Replicate(HU, 3) )
        -- B, M, HU
        return md
    end

    -- the sub nets
    local parNet = nn.ParallelTable()
        :add( make_cv(KH) )
        :add( make_tfidf() )

    -- net
    local md = nn.Sequential()

    -- {B, M (,V)}, {B, M (,V)}
    md:add(parNet)
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
        print( ('reinit params uniform, [%4.3f, %4.3f]'):format(-b,b) )

        local params, _ = md:getParameters()
        params:uniform(-b,b)
    end
    reinit_params(md)

    local function md_reset(md, arg)
        -- fine to do nothing
    end

    return md, md_reset
end

return this