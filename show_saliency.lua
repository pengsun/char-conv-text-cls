require'cudnn'
require'OneHot'
util = require'util.misc'

local envFn = 'seqLength40-batSize250-HU190-conv6x1_conv4x1_conv2x1_convSeqLenx1_conv1x1_epoch200.00_lossval0.1176.t7'

math.randomseed(os.time())
local opt = {
    dataPath = 'data/ptb-charwordwin-pos.lua',
    dataMask = {tr=false, val=false, te=true},

    envPath = path.join('cv', 'ptb-charwordwin2d-randbeg-pos', envFn),

    seqLength = 887,
    batSize = 1,

    --ibat = 1293, -- which data batch interested
    --iclass = nil, -- which class interested, use that from label y if nil value
    ibat = math.random(1,25000), -- which data batch interested
    iclass = nil, -- which class interested, use that from label y if nil value
}

local function load_data(opt)
    local tr,val,te,vchar = dofile(opt.dataPath)
    local loader = te
    return loader, vchar
end

local function load_md(opt)
    local fn = opt.envPath or "/env/path/not/specified"
    print('loading model from ' .. fn)
    local env = torch.load(fn)

    local md, md_reset = env.md, env.md_reset
    local cri = env.cri or nn.ClassNLLCriterion()

    -- reset batch size and sequence length
    assert(opt.seqLength==env.opt.seqLength,
        string.format("opt.segLength (=%d) must be equal to that of the saved model", opt.seqLength)
    )
    md = md_reset(md, opt)
    print(md)

    return md, md_reset, cri
end

local function ten_to_tablechars(x, ivocab)
    local tchar = {}

    assert(x:dim()==2)
    local sz = x:size()
    for i = 1, sz[1] do
        local str = ""
        for j = 1, sz[2] do
            local c = ivocab[x[i][j]] or '_' -- the char or null as space
            if c == "\n" then
                error('%d, %d: encounter newline!'):format(i,j)
            end
            str = str .. c
        end
        tchar[i] = str
    end
    return tchar
end

local function get_data_batch(loader, ibat)
    print('getting data batch ' .. ibat)
    for i = 1, ibat-1 do
        loader:next_batch()
    end
    return loader:next_batch()
end

local function tchars_saliency_to_str(tchars, s)
    assert(#tchars == s:size(1))

    local str = ""
    for i = 1, #tchars do
        -- chars line followed by saliency score line
        str = str .. tchars[i] .. "\n"
        for j = 1, s:size(2) do
            str = str .. s[i][j]
        end
        str = str .. "\n"
    end
    return str
end

local function normalize_01(s)
    --- normalize to [0,1]
    local m = s:max()
    local margin = m/9
    s = s:div(margin):floor()
    s = s:byte()
    return s
end

local function calc_saliency(md, x, iclass)
    x = x:cuda(); md:cuda();

    --- fprop
    local outputs = md:forward(x)

    --- bprop
    local function make_gradOutputs()
        local B, K = outputs:size(1), outputs:size(2)
        local go = torch.zeros(B,K, 'torch.CudaTensor')
        go[{ {}, {iclass} }]:fill(1)

        return go
    end
    local gradOutputs = make_gradOutputs()
    md:backward(x, gradOutputs)

    --- get the abs grad map, max over channel V
    local imodule = 2
    local g = md:get(imodule).gradInput -- B,M,V
    local B, M = g:size(1), g:size(2)
    g = g:abs():max(3):reshape(B,M) -- B, M

    g = g:float()
    return normalize_01(g)
end

local function main(opt)
    local opt = opt or {}
    print('options:')
    print(opt)

    --- load data, model
    local loader, vocabChar = load_data(opt)
    local ivocabChar = util.make_inverse_vocabulary(vocabChar)
    local md, md_reset, cri = load_md(opt)

    --- get data batch
    local ibat = opt.ibat or 1
    local x, y = get_data_batch(loader, ibat)

    --- get input chars,
    local tchars = ten_to_tablechars(x, ivocabChar) -- B, {M}

    --- get saliency map
    local function get_iclass()
        local iclass = opt.iclass
        if not iclass then --- use the class from label y
            local iInst = math.ceil(x:size(1)/2)
            iclass = y[iInst]
        end
        return iclass
    end
    local iclass = get_iclass()
    print('calculating saliency map for class ' .. iclass .. ': ' .. iclass)
    local s = calc_saliency(md, x, iclass) -- B, M

    --- render it
    local str = tchars_saliency_to_str(tchars, s)
    print("\n")
    print(str)

end

main()
