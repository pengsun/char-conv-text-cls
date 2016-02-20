require'cudnn'
require'OneHot'
util = require'util.misc'

local function load_data(opt)
    local tr,val,te,vchar = dofile(opt.dataPath).main(opt)
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

local function tensor_to_chars(x, ivocab)
    local tchar = {}

    assert(x:dim()==2 and x:size(1)==1)

    local str = ""
    for i = 1, x:numel() do
        local c = ivocab[x[1][i]] or '_' -- the char or null as space
        if c == "\n" then
            error('%d: encounter newline!'):format(i)
        end
        str = str .. c
    end
    return str
end

local function get_data_batch(loader, ibat)
    print('getting data batch ' .. ibat)
    loader:evaluate()
    for i = 1, ibat-1 do
        loader:next_batch()
    end
    return loader:next_batch()
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

    local function normalize_01(s)
        --- normalize to [0,1]
        local m = s:max()
        local margin = m/9
        s = s:div(margin):floor()
        s = s:byte()
        return s
    end

    return normalize_01(g), outputs
end

local function chars_saliency_to_str(chars, s, dispWidth)
    dispWidth = dispWidth or 50

    s = s:squeeze()
    assert(#chars == s:numel())

    local str = ""
    local linelen = 0
    for i = 1, #chars do
        str = str .. string.sub(chars, i,i)
        linelen = linelen + 1

        if i%dispWidth == 0 or i == #chars then
            -- append newline for string
            str = str .. "\n"

            -- append score string
            for j = i-linelen+1, i do
                str = str .. s[j]
            end
            -- append newline for score string
            str = str .. "\n"
            -- clear
            linelen = 0
        end
    end
    return str
end

local this = {}
this.main = function(opt)
    local function get_dft_opt()
        local envFn = 'seqLength887-HU1000-cv8max-o_epoch15.00_lossval0.3137.t7'

        math.randomseed(os.time())
        local opt = {
            dispWidth = 210,

            dataPath = 'data/imdb-fix.lua',
            dataMask = {tr=false, val=false, te=true},

            envPath = path.join('cv', 'imdb-randchar', envFn),

            seqLength = 887,
            batSize = 1,

            ibat = 24837, -- which data batch interested
            iclass = 1, -- which class interested, use that from label y if nil value

            --    ibat = math.random(1,25000), -- which data batch interested
            --    iclass = nil, -- which class interested, use that from label y if nil value
        }
        return opt
    end

    local opt = opt or get_dft_opt()
    print('options:')
    print(opt)

    --- load data, model
    assert(opt.batSize==1, "batSize must equal to 1")
    local loader, vocabChar = load_data(opt)
    local ivocabChar = util.make_inverse_vocabulary(vocabChar)
    local md, md_reset, cri = load_md(opt)

    --- get data batch
    local ibat = opt.ibat or 1
    local x, y = get_data_batch(loader, ibat)
    assert(y:numel()==1)
    print('target = ' .. y[1])

    --- get input chars
    local chars = tensor_to_chars(x, ivocabChar) -- B, {M}

    --- get saliency map
    local function get_iclass()
        local iclass = opt.iclass or y[1]
        return iclass
    end
    local iclass = get_iclass()
    print('calculating saliency map for class ' .. iclass)
    local s, outputs = calc_saliency(md, x, iclass) -- B, M

    --- show prediction
    print('outputs = ' .. outputs[1][1], ', ' .. outputs[1][2])
    local _, pred = outputs:max(2)
    print('prediction = ' .. pred:squeeze())

    --- render the sailiency map
    local dispWidth = opt.dispWidth or 70
    local str = chars_saliency_to_str(chars, s, dispWidth)
    print("\n")
    print(str)

end

return this
