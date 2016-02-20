require'cudnn'
require'OneHot'
util = require'util.vocab'

-- helper
local function load_data(opt)
    local tr,val,te,vchar = dofile(opt.dataPath).main(opt)
    local loader = te
    return loader, vchar
end

local function load_md(opt)
    local fn = opt.envPath or error('no envPath')
    print('loading model from ' .. fn)
    local env = torch.load(fn)

    local md, md_reset = env.md, env.md_reset
    local cri = env.cri or nn.ClassNLLCriterion()

    -- reset batch size and sequence length
    md_reset(md, opt)
    print(md)

    return md, md_reset, cri
end

local function tensor_to_tokens(x, ivocab)
    assert(x:dim()==2 and x:size(1)==1, "wrong x size. only support batSize = 1")

    local str = {}
    for i = 1, x:numel() do
        local w = ivocab[x[1][i]] or '<unknown>' -- the char or null as space
        if w == "\n" then
            error('%d: encounter newline!'):format(i)
        end
        table.insert(str, w)
    end
    return str
end

local function get_data_batch(loader, ibat)
    loader:cuda()

    print(loader)
    print('getting data batch ' .. ibat)

    for i = 1, ibat-1 do
        loader:next_batch()
    end
    return loader:next_batch()
end

-- saliency calculation
local function normalize_score(s, maxTarVal)
    local maxval = maxTarVal or 1
    local m = s:max()
    local margin = m/maxTarVal
    s = s:div(margin):floor()
    s = s:byte()
    return s
end

local function calc_saliency(md, x, iclass)
    --x = x:cuda()
    --md:cuda();

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
    -- enable grad input
    local mods = md:findModules('nn.OneHotTemporalConvolution')
    for j, mod in ipairs(mods) do
        mod:should_updateGradInput(true)
    end
    -- do the real bprop
    md:backward(x, gradOutputs)

    --- get the abs grad map, max over channel V
    local iModoulePeek = 1
    local g = md:get(iModoulePeek).gradInput -- B,M,V
    local B, M = g:size(1), g:size(2)
    g = g:abs():max(3):reshape(B,M) -- B, M

    g = g:float() -- B, M

    return g, outputs
end

-- renderer
local function render_print(tokens, s)
    local function words_saliency_to_str(words, s)
        s = s:view(-1)
        assert(#words==s:numel())

        local str = ""
        for i = 1, #words do
            str = str .. words[i] .. "  " .. s[i] .. "\n"
        end

        return str
    end

    local str = words_saliency_to_str(tokens, s)
    print("\n")
    print(str)
end

local function render_html(words, s)
    local function wordval_to_item(word, v)
        local tmpl = [[<span style="color:#%s0000">%s</span>]]
        local strColor = ('%02x'):format(v):upper()
        return tmpl:format(strColor, word)
    end

    local function words_saliency_to_htmlstr(words, s)
        local header = [[<p style="font-size:12px;">]] .. "\n"
        local tail = [[</p>]]
        local content = ""
        for i, word in ipairs(words) do
            content = content ..
                    wordval_to_item(word, s[i]) ..
                    " "
        end
        return header .. content .. tail
    end

    require'mobdebug'.start()
    s = s:view(-1) -- vectorized
    assert(#words == s:numel())

    local str = words_saliency_to_htmlstr(words, s)
    file = require'pl.file'
    local fn = path.tmpname() .. '.html'
    file.write(fn, str)
    os.execute('xdg-open ' .. fn)
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

            renderer = 'print'
        }
        return opt
    end

    local opt = opt or get_dft_opt()
    print('options:')
    print(opt)

    --- load data, model
    assert(opt.batSize==1, "batSize must equal to 1")
    local loader, vocabWord = load_data(opt)
    local ivocabWord = util.make_inverse_vocabulary(vocabWord)
    local md, md_reset, cri = load_md(opt)

    --- get data batch
    local ibat = opt.ibat or 1
    local x, y = get_data_batch(loader, ibat)
    assert(y:numel()==1)
    print('target = ' .. y[1])

    --- get input tokens as string table
    --require'mobdebug'.start()
    local tokens = tensor_to_tokens(x, ivocabWord)

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
    if opt.renderer == 'print' then
        local maxTarVal = 9
        render_print(tokens, normalize_score(s,maxTarVal))
    elseif opt.renderer == 'html' then
        local maxTarVal = 255
        render_html(tokens, normalize_score(s,maxTarVal))
    else
        error('unknown renderer ' .. opt.renderer)
    end

end

return this
