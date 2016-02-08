require'nn'
require'cudnn'
require'OneHot'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 300 -- vocabulary/embedding size
    local M = opt.seqLength or 291
    local HU = opt.HU or 190

    local kH = 3
    local pad = 1
    local pool = 3

    local get_output_size = function(Min)
        return math.floor( Min/pool )
    end
    local Md = get_output_size(M)
    local Mdd = get_output_size(Md)

    local md = nn.Sequential()
    -- B, M, V
    md:add( nn.Transpose({2,3}, {1,2}) )
    -- V, B, M
    md:add(nn.Reshape(1,V,B,M, false)) -- 2
    -- 1, V, B, M

    -- 1, V, B, M
    md:add(cudnn.SpatialConvolution(V, HU, kH, 1, 1,1, pad,0))
    md:add(cudnn.ReLU(true))
    -- 1, HU, B, M
    md:add(cudnn.SpatialMaxPooling(pool, 1))
    md:add(nn.Dropout())
    -- 1, HU, B, M'

    -- 1, HU, B, M'
    md:add( cudnn.SpatialConvolution(HU, HU, kH, 1, 1,1, pad,0) )
    md:add(cudnn.ReLU(true))
    -- 1, HU, B, M'
    md:add(cudnn.SpatialMaxPooling(pool, 1))
    md:add(nn.Dropout())
    -- 1, HU, B, M''

    -- 1, HU, B, M''
    md:add( cudnn.SpatialConvolution(HU, HU, kH, 1, 1,1, pad,0) )
    md:add(cudnn.ReLU(true))
    -- 1, HU, B, M''
    md:add(cudnn.SpatialMaxPooling(Mdd, 1)) -- 13
    md:add(nn.Dropout())
    -- 1, HU, B, 1

    -- 1, HU, B, 1
    md:add(cudnn.SpatialConvolution(HU, K, 1, 1))
    -- 1, K, B, 1

    -- 1, K, B, 1
    md:add(nn.Reshape(K, B, false)) -- 16
    -- K, B
    md:add(nn.Transpose{1,2})
    -- B, K
    md:add(cudnn.LogSoftMax())
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print('reinit params normal, std = ' .. b)

        local params, _ = md:getParameters()
        params:normal(0,b)
    end
    reinit_params(md)

    local function md_reset(md, arg)
        local B = arg.batSize or error('no batSize')
        local M = arg.seqLength or error('no seqLength')

        -- input reshape
        md.modules[2].size[3] = B
        md.modules[2].size[4] = M
        -- maxpooling
        local Md = get_output_size(M)
        local Mdd = get_output_size(Md)
        md.modules[13].kW = Mdd
        -- output reshape
        md.modules[16].size[2] = B

        return md
    end

    return md, md_reset
end

return this

