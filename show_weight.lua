require'cudnn'
require'OneHot'
util = require'util.vocab'
sa = require'show_saliency_word'

local function normalize_score(s)
    local m = s:min()
    local M = s:max()
    local d = math.max(M - m, 1e-6)
    -- (s-m)/(M-m)
    local rs = s:add(-m):div(d)
    return rs
end

local function extract_weight(md)
    --require'mobdebug'.start()
    -- the controller
    local iconcat = 1
    local ic = 2
    local con = md:get(iconcat):get(ic)
    -- the weight
    local iw = con:size() - 1 -- 2nd last
    local m = con:get(iw)
    local w = m.output:clone()
    -- B, M
    return w
end

local function calc_weight(md, x)
    x = x:cuda()
    md:cuda()
    md:evaluate()

    local outputs = md:forward(x) -- B, K
    local g = extract_weight(md) -- B, M
    return g, outputs
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

            saType = 'pos',
            renderer = 'print',
        }
        return opt
    end

    local opt = opt or get_dft_opt()
    print('options:')
    print(opt)

    --- load data, model
    assert(opt.batSize==1, "batSize must equal to 1")
    local loader, vocabWord = sa.load_data(opt)
    local ivocabWord = util.make_inverse_vocabulary(vocabWord)
    local md, md_reset, cri = sa.load_md(opt)

    --- get data batch
    local ibat = opt.ibat or 1
    local x, y = sa.get_data_batch(loader, ibat)
    assert(y:numel()==1)
    print('target = ' .. y[1])

    --- get input tokens as string table
    local tokens = sa.tensor_to_tokens(x, ivocabWord)

    --- get weights
    print('getting weights...')
    local s, outputs = calc_weight(md, x) -- B, M; B, K

    --- show prediction
    print('outputs = ' .. outputs[1][1], ', ' .. outputs[1][2])
    local _, pred = outputs:max(2)
    print('prediction = ' .. pred:squeeze())

    --- render the sailiency map
    if opt.renderer == 'print' then
        --local rs = normalize_score(s)
        rs = s
        rs = rs:mul(100):round()
        sa.render_print(tokens, rs)
    elseif opt.renderer == 'html' then
        local rs = normalize_score(s)
        rs = rs:mul(255)
        sa.render_html(tokens, rs)
    else
        error('unknown renderer ' .. opt.renderer)
    end
end

return this