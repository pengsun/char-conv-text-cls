require 'torch'
require 'nn'
require 'cudnn'
require 'optim'

require 'pl.path'
require 'pl.tablex'
require 'xlua'

require 'OneHot'
local utmisc = require 'util.misc'

local function init_env()
    local env = {
        i = 0,
        epoch = nil,
        lossesTr = {},
        lossesVal = {},
        errVal = {},
        md = nil,
    }
    return env
end

local function init_global(opt)
    -- gpu
    if opt.gpuId > 0 then
        require('cutorch').setDevice(opt.gpuId)
    end

    -- random number seed
    local seed = opt.seed or 123
    torch.manualSeed(seed)
    if opt.gpuId > 0 then
        cutorch.manualSeedAll(seed)
    end
    math.randomseed(seed)

    -- ensure environment saving path
    if not path.isdir(opt.envSavePath) then
        local ok, info = path.mkdir(opt.envSavePath)
        if not ok then
            error("cannot create env saving path " .. opt.envSavePath .. " " .. info)
        end
    end

    -- redirect print
    local f = assert(io.open(opt.logSavePath, 'w')) -- fine not to close it
    print = utmisc.get_print_screen_and_file(f)
end

local function get_dft_opt()
    local function make_lrEpCheckpoint()
        local baseRate, factor = 1e-2, 0.98
        local r = {}
        for i = 1, 10 do
            r[i] = baseRate
        end
        for i = 11, 100 do
            r[i] = r[i - 1] * factor
        end
        return r
    end

    local opt = {}

    opt.gpuId = 1
    opt.seed = 123

    opt.dataPath = 'data/ptb.lua'
    opt.dataMask = { tr = true, val = true, te = true }

    opt.mdPath = 'net/convOneHot.lua'
    opt.criPath = 'net/cri-nll-one.lua'

    opt.envSavePrefix = 'convOneHot'
    opt.envSavePath = 'cv/ptb'
    opt.envContinuePath = ''

    opt.logSavePath = 'zzz.log'

    opt.V = 83
    opt.HU = 190

    opt.maxEp = 10
    opt.batSize = 250
    opt.printFreq = 1
    opt.evalFreq = 10
    opt.showEpTime = true -- show eopch time
    opt.showIterTime = true -- show iteration time

    --opt.gradClamp = 5
    opt.lrEpCheckpoint = make_lrEpCheckpoint()

    opt.optimMethod = optim.rmsprop
    opt.optimState = {
        learningRate = 2e-3,
        alpha = 0.95, -- decay rate
    }
    opt.weightDecayOutputLayer = 0

    return opt
end

local this = {}

this.main = function(opt)
    -- parse option, do global settings
    opt = opt or {}
    opt = tablex.merge(get_dft_opt(), opt, true)
    init_global(opt)

    print('[options]')
    print(opt)

    print('[training environment]')
    local env
    if #opt.envContinuePath > 0 then
        print('load from file ' .. opt.envContinuePath)
        env = torch.load(opt.envContinuePath)

        local function overwrite_opt()
            local lrEpCheckpoint = opt.lrEpCheckpoint
            local maxEp = opt.maxEp
            local envSavePath = opt.envSavePath
            local printFreq = opt.printFreq
            local evalFreq = opt.evalFreq

            opt = env.opt
            opt.maxEp = maxEp
            opt.lrEpCheckpoint = lrEpCheckpoint
            opt.envSavePath = envSavePath
            opt.printFreq = printFreq
            opt.evalFreq = evalFreq
        end

        overwrite_opt()

        -- print new
        print('[updated option]')
        print(opt)
    else
        print('init from scratch')
        env = init_env()
    end

    print('[data]')
    local loaderTr, loaderVal, loaderTe = dofile(opt.dataPath).main(opt)

    print('[model]')
    local md, md_reset
    if env.md and env.md_reset then
        md, md_reset = env.md, env.md_reset
    else
        md, md_reset = dofile(opt.mdPath).main(opt)
    end
    local cri = env.cri or dofile(opt.criPath).main(opt)
    print(md)

    print('[training logic]')
    local function to_gpu()
        if opt.gpuId > 0 then -- to gpu
        require 'cutorch'
        require 'cunn'
        loaderTr:cuda();
        if loaderVal then loaderVal:cuda() end
        if loaderTe then loaderTe:cuda() end -- lazy
        md:cuda(); cri:cuda();
        end
    end

    to_gpu()

    -- prepare loader & data
    local nb = loaderTr:num_batches() -- rewind to ensure opt.batSize
    local maxIt = nb * opt.maxEp
    local begIt = env.i or 0
    print('#batch size = ' .. opt.batSize)
    print('#batches per epoch = ' .. nb)
    print('#max epochs = ' .. opt.maxEp)
    print('initial iteration = ' .. begIt)

    -- prepare loss
    local lossesTr = env.lossesTr or {}
    local lossesVal = env.lossesVal or {}
    local errVal = env.errVal or {}

    -- prepare model
    md:training() -- for dropout

    local params, gradParams = md:getParameters()
    print('#parameters = ' .. params:numel())

    local function getParametersOutputLayer()
        local pp, gg = md:parameters()
        local n = #gg
        assert(n > 2, "too few parameter layers")
        local weight, bias = pp[n-1], pp[n]
        local gWeight, gBias = gg[n-1], gg[n]
        return weight, bias, gWeight, gBias
    end
    local weightOL, biasOL, gWeightOL, gBiasOL = getParametersOutputLayer()

    -- iterate over batches
    for i = begIt + 1, maxIt do
        local epoch = i / nb
        local timeIter, timeIterData = 0, 0
        local curBatSize

        local function eval_dataset(loader)
            loader:reset_batch_pointer()
            local lossTotal = 0
            local numClasses = opt.numClasses or error('no opt.numClasses!')
            local conf = optim.ConfusionMatrix(numClasses) -- confusion matrix

            print('evaluate on the dataset')
            local nb = loader:num_batches()
            local numEval = 0
            for ibat = 1, nb do
                -- data batch
                local inputs, targets = loader:next_batch()

                -- fprop
                md:evaluate()
                local outputs = md:forward(inputs)
                local loss = cri:forward(outputs, targets)

                -- update loss
                numEval = numEval + targets:numel()
                lossTotal = lossTotal + loss * targets:numel()

                -- update error
                local predictions = outputs
                if type(outputs) == 'table' then -- last one as the predictions
                    predictions = outputs[#outputs]
                end
                conf:batchAdd(predictions, targets)

                xlua.progress(ibat, nb)
            end
            local lossAvg = lossTotal / numEval
            print('evaluation average loss = ' .. lossAvg)

            conf:updateValids()
            print('evaluation error rate = ' .. 1 - conf.totalValid)

            md:training() -- restore
            return lossAvg, 1 - conf.totalValid
        end

        local function feval(pp)
            if pp ~= params then params:copy(pp) end
            gradParams:zero()

            -- get data batch
            local timeData
            -----------------------------------------------------
            if opt.showIterTime then timeData = torch.tic() end
            local inputs, targets = loaderTr:next_batch()
            if opt.showIterTime then
                timeData = torch.toc(timeData)
                timeIterData = timeIterData + timeData
            end
            -----------------------------------------------------
            curBatSize = opt.batSize -- TODO: what if variable batch?

            -- fprop
            local outputs = md:forward(inputs)
            local loss = cri:forward(outputs, targets)

            -- bprop
            local gradOutputs = cri:backward(outputs, targets)
            md:backward(inputs, gradOutputs)

            -- regularization
            --gradParams:clamp(-opt.gradClamp, opt.gradClamp) -- grad clampping
            local wdol = opt.weightDecayOutputLayer -- output layer L2 regularizer
            if wdol > 0 then
                gWeightOL:add(wdol, weightOL)
                --gBiasOL:add(wdol, biasOL)
            end

            return loss, gradParams
        end

        local function print_progress()
            if opt.showIterTime == true then
                local tmpl = '%d/%d (epoch %.3f), ' ..
                        'train_loss = %6.8f, grad/param norm = %6.4e, ' ..
                        'speed = %5.1f/s, %5.3fs/iter, data_load = %4.2f%%'
                print(string.format(tmpl,
                    i, maxIt, epoch,
                    lossesTr[i], gradParams:norm() / params:norm(),
                    curBatSize / timeIter, timeIter, timeIterData / timeIter))
            else
                local tmpl = '%d/%d (epoch %.3f), ' ..
                        'train_loss = %6.8f, grad/param norm = %6.4e'
                print(string.format(tmpl,
                    i, maxIt, epoch,
                    lossesTr[i], gradParams:norm() / params:norm()))
            end
        end

        local function change_lr_when_available()
            local int_epoch = math.ceil(epoch)
            if opt.lrEpCheckpoint[int_epoch] then
                print('iter: ' .. i .. ', change to learning rate: ' .. opt.lrEpCheckpoint[int_epoch])
                opt.optimState.learningRate = opt.lrEpCheckpoint[int_epoch]
            end
        end

        local function save_env(env)
            -- remove intermediate data
            utmisc.cleanup_model(env.md)

            local fn = string.format('%s_epoch%.2f_lossval%.4f_errval%1.2f.t7',
                opt.envSavePrefix, env.epoch, env.lossesVal[env.i], 100 * env.errVal[env.i])
            local ffn = path.join(opt.envSavePath, fn)

            print('saving to ' .. ffn)
            torch.save(ffn, env)
        end

        -- epoch begining
        if (i % nb) == 1 then
            if opt.showEpTime == true then
                print("time now = " .. utmisc.get_current_time_str())
            end

            change_lr_when_available()

            -- shuffle data at epoch beginning
            --loaderTr:reset_batch_pointer()
        end

        -- do the optimization
        -----------------------------------------------------------
        if opt.showIterTime == true then
            timeIter = torch.tic()
        end
        local _, lst = opt.optimMethod(feval, params, opt.optimState)
        local loss = lst[1]
        if opt.showIterTime == true then
            if opt.gpuId > 0 then cutorch.synchronize() end
            timeIter = torch.toc(timeIter)
        end
        -----------------------------------------------------------

        -- update
        lossesTr[i] = loss

        -- print?
        if i % opt.printFreq == 0 then
            print_progress()
        end

        -- evaluate (on validation set ?), update and save
        if i % opt.evalFreq == 0 then
            lossesVal[i], errVal[i] = eval_dataset(loaderVal)

            save_env {
                opt = opt,
                i = i,
                epoch = epoch,
                lossesTr = lossesTr,
                lossesVal = lossesVal,
                errVal = errVal,
                md = md,
                cri = cri,
                md_reset = md_reset,
            }
        end

        if i % 10 == 0 then collectgarbage() end
    end -- for i

    return env
end -- main

return this