require'nn'
require'cudnn'
require'OneHot'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    local opt = opt or {}
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 82 -- vocabulary size
    local M = opt.seqLength or 512
    local HU = opt.HU or 190

    local kH1, kH2, kH3 = 6, 4, 2 -- conv kernel size
    local pool = 2

    local Md = math.floor( (M-kH1+1) / pool)
    local Mdd = math.floor( (Md-kH2+1) / pool)
    local Mddd = math.floor( (Mdd-kH3+1) / (2*pool) )

    local md = nn.Sequential()
    -- B, M
    md:add(OneHot(V))
    -- B, M, V
    md:add( nn.Transpose({2,3}, {1,2}) )
    -- V, B, M
    md:add(nn.Reshape(1,V,B,M, false)) -- 3
    -- 1, V, B, M

    -- 1, V, B, M
    md:add(cudnn.SpatialConvolution(V, HU, kH1,1))
    md:add(cudnn.ReLU(true))
    md:add(cudnn.SpatialMaxPooling(pool, 1))
    md:add(nn.Dropout())
    -- 1, HU, B, M'

    -- 1, HU, B, M'
    md:add(cudnn.SpatialConvolution(HU, HU, kH2,1))
    md:add(cudnn.ReLU(true))
    md:add(cudnn.SpatialMaxPooling(pool, 1))
    md:add(nn.Dropout())
    -- 1, HU, B, M''

    -- 1, HU, B, M''
    md:add( cudnn.SpatialConvolution(HU, HU, kH3,1) )
    md:add(cudnn.ReLU(true))
    md:add(cudnn.SpatialMaxPooling(2*pool, 1))
    md:add(nn.Dropout())
    -- 1, HU, B, M'''

    -- 1, HU, B, M''' -- FC
    md:add(nn.Dropout())
    md:add(cudnn.SpatialConvolution(HU, HU, Mddd, 1))
    md:add(cudnn.ReLU(true))
    md:add(nn.Dropout())
    -- 1, HU, B, 1

    -- 1, HU, B, 1 -- output
    md:add(cudnn.SpatialConvolution(HU, K, 1, 1))
    -- 1, K, B, 1
    md:add(nn.Reshape(K, B, false)) -- 21
    -- K, B
    md:add(nn.Transpose{1,2})
    -- B, K
    md:add(cudnn.LogSoftMax())
    -- B, K

    local function reinit_params_uniform(md)
        local b = opt.paramInitBound or 0.08
        print('reinit params uniform [' .. -b .. ', ' .. b ..']')

        local params, _ = md:getParameters()
        params:uniform(-b, b)
    end
    local function reinit_params_norm(md)
        local b = opt.paramInitBound or 0.08
        print('reinit params normal, std = ' .. b)

        local params, _ = md:getParameters()
        params:normal(0,b)
    end
    reinit_params_uniform(md)

    local function md_reset(md, arg)
        local B = arg.batSize or error('no batSize')

        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsistent seqLength M")

        -- input reshape
        md.modules[3].size[3] = B

        -- output reshape
        md.modules[21].size[2] = B
    end

    return md, md_reset
end

return this

