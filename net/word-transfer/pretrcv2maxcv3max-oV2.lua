require'nn'
require'cudnn'
require'onehot-temp-conv'
utv = require'util.vocab'

--cudnn.fastest = true

local this = {}

this.main = function(opt)
    -- parse option
    local opt = opt or {}
    -- net size
    local K = opt.numClasses or 2 -- #classes
    local B = opt.batSize or 16 -- batch size
    local V = opt.V or 300 -- vocabulary/embedding size
    local M = opt.seqLength or 291
    local HU = opt.HU or 190
    -- pre-trained model
    local fnVocabThis = assert(opt.fnVocabThis)
    local fnVocabThat = assert(opt.fnVocabThat)
    local fnEnvThat = assert(opt.fnEnvThat)

    -- convbank this
    local kH1, kH2 = 2, 3

    local function make_cvmax(kH)
        local md = nn.Sequential()
        -- B, M (,V)
        md:add( nn.OneHotTemporalConvolution(V, HU, kH) )
        -- B, M-kH1+1, HU
        md:add( nn.TemporalMaxPooling(M-kH+1) )
        -- B, 1, HU
        return md
    end

    local function make_convmaxbank ()
        local ct = nn.ConcatTable()
        ct:add( make_cvmax(kH1) )
        ct:add( make_cvmax(kH2) )
        return ct
    end

    local convbankThis = make_convmaxbank()

    -- the model
    local md = nn.Sequential()

    -- B, M (,V)
    md:add( convbankThis )
    -- {B, 1, HU}, {B, 1, HU}
    md:add( nn.JoinTable(3, 3) )
    -- B, 1, 2*HU
    md:add( nn.Reshape(2*HU, true) )
    -- B, 2*HU
    md:add( nn.ReLU(true) )
    md:add( nn.Dropout() )
    -- B, 2*HU

    -- B, 2*HU
    md:add( nn.Linear(2*HU, K) )
    -- B, K
    md:add( cudnn.LogSoftMax() )
    -- B, K

    local function reinit_params(md)
        local b = opt.paramInitBound or 0.08
        print( ('reinit params uniform, [%4.3f, %4.3f]'):format(-b,b) )

        local params, _ = md:getParameters()
        params:uniform(-b,b)
        print('params norm = ' .. params:norm())
    end
    reinit_params(md)

    --require'mobdebug'.start()
    print('reinit part of the weights with pretrained model...')
    local function get_common_vocab(fnVocabThis, fnVocabThat)
        local vthis, vthat = torch.load(fnVocabThis), torch.load(fnVocabThat)
        local vc, idxThis, idxThat = utv.get_common_vocab(vthis, vthat)

        print('common vocab size = ' .. tablex.size(vc) ..
                ", vocab this size = " ..tablex.size(vthis) ..
                ", vocab that size = " .. tablex.size(vthat))

        local vocabIdxThis = torch.LongTensor(idxThis)
        local vocabIdxThat = torch.LongTensor(idxThat)
        return vc, vocabIdxThis, vocabIdxThat
    end
    local dummy, vocabIdxThis, vocabIdxThat = get_common_vocab(fnVocabThis, fnVocabThat)

    local function load_convbankThat(fnEnvThat)
        print('loading that model from ' .. fnEnvThat)
        local envThat = torch.load(fnEnvThat)
        local mdThat = envThat.md
        return mdThat:get(1) -- module 1
    end
    local convbankThat = load_convbankThat(fnEnvThat)

    local function init_convbankThis_with_convbankThat(convbankThis, convbankThat, vocabIdxThis, vocabIdxThat)
        print('using weights from that model...')
        local numConvmax = #convbankThis.modules
        assert(numConvmax == #convbankThat.modules, "different #convmax")

        for i = 1, numConvmax do
            local convThis = convbankThis:get(i):get(1)
            local convThat = convbankThat:get(i):get(1)

            --require'mobdebug'.start()
            convThis:index_copy_weight(vocabIdxThis, convThat,vocabIdxThat)
        end
    end
    init_convbankThis_with_convbankThat(convbankThis, convbankThat, vocabIdxThis, vocabIdxThat)
    local params, _ = md:getParameters()
    print('params norm = ' .. params:norm())

    local function md_reset(md, arg)
        local newM = arg.seqLength or error('no seqLength')
        assert(newM==M, "inconsisten seq length")
    end

    --require'mobdebug'.start()
    collectgarbage()
    return md, md_reset
end

return this