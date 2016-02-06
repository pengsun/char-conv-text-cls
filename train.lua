require'torch'
require'nn'
require'cudnn'
require'optim'

require'pl.path'
require'pl.tablex'
require'xlua'

require'OneHot'
local misc = require'util.misc'

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

local function setup_global(opt)
  -- gpu
  if opt.gpuId > 0 then
    require('cutorch').setDevice(opt.gpuId)
  end

  -- random number seed
  torch.manualSeed(opt.seed or 123)

  -- ensure environment saving path
  if not path.isdir(opt.envSavePath) then
    local ok, info = path.mkdir(opt.envSavePath)
    if not ok then
      error("cannot create env saving path " .. opt.envSavePath .. " " .. info)
    end
  end
  
end

local function get_dft_opt()
  local function make_lrEpCheckpoint()
    local baseRate, factor = 1e-2, 0.98
    local r = {}
    for i = 1, 10 do
      r[i] = baseRate
    end
    for i = 11, 100 do
      r[i] = r[i-1] * factor
    end
    return r
  end

  local opt = {}
  
  opt.gpuId = 1
  opt.seed = 123

  opt.dataPath = 'data/ptb.lua'
  opt.dataMask = {tr=true,val=true,te=true}
  
  opt.mdPath = 'net/convOneHot.lua'

  opt.envSavePrefix = 'convOneHot'
  opt.envSavePath = 'cv/ptb'
  opt.envContinuePath = ''

  opt.V = 83
  opt.HU = 190

  opt.maxEp = 10
  opt.batSize = 250
  opt.printFreq = 1
  opt.evalFreq = 10

  opt.gradClamp = 5
  opt.lrEpCheckpoint = make_lrEpCheckpoint()

  opt.optimState = {
    learningRate = 2e-3, 
    alpha = 0.95, -- decay rate
  }
  
  return opt
end

local this = {}

this.main = function (opt)
  print('[options]')
  opt = opt or {}
  opt = tablex.merge(get_dft_opt(), opt, true)
  print(opt)
  setup_global(opt)

  print('[training environment]')
  local env
  if #opt.envContinuePath > 0 then
    print('load from file ' .. opt.envContinuePath)
    env = torch.load( opt.envContinuePath )

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
  local cri = env.cri or nn.ClassNLLCriterion()
  print(md)
  
  print('[training logic]')
  local function to_gpu()
    if opt.gpuId > 0 then -- to gpu
      require'cutorch'
      require'cunn'
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

  -- iterate over batches
  for i = begIt+1, maxIt do
    local epoch = i/nb
    local time
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
        lossTotal = lossTotal + loss*targets:numel()
        -- update error
        conf:batchAdd(outputs, targets)

        xlua.progress(ibat, nb)
      end
      local lossAvg = lossTotal/numEval
      print('evaluation average loss = ' .. lossAvg)

      conf:updateValids()
      print('evaluation error rate = ' .. 1-conf.totalValid)

      md:training() -- restore
      return lossAvg, 1-conf.totalValid
    end

    local function feval(pp)
      if pp ~= params then params:copy(pp) end
      gradParams:zero()

      -- get data batch
      local inputs, targets = loaderTr:next_batch()
      curBatSize = inputs:size(1)

      -- fprop
      local outputs = md:forward(inputs)
      local loss = cri:forward(outputs, targets)

      -- bprop
      local gradOutputs = cri:backward(outputs, targets)
      md:backward(inputs, gradOutputs)

      -- regularization
      gradParams:clamp(-opt.gradClamp, opt.gradClamp)
      
      return loss, gradParams
    end
    
    local function print_progress()
      local tmpl = '%d/%d (epoch %.3f), ' ..
        'train_loss = %6.8f, grad/param norm = %6.4e, ' ..
        'speed = %5.1f/s, %5.3fs/iter'
      print(string.format(tmpl,
        i, maxIt, epoch,
        lossesTr[i], gradParams:norm() / params:norm(),
        curBatSize/time, time))
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
      misc.cleanup_model(env.md)

      local fn = string.format('%s_epoch%.2f_lossval%.4f_errval%1.2f.t7',
        opt.envSavePrefix, env.epoch, env.lossesVal[env.i], 100*env.errVal[env.i])
      local ffn = path.join(opt.envSavePath, fn)

      print('saving to ' .. ffn)
      torch.save(ffn, env)
    end

    -- epoch begining
    if (i%nb) == 1 then
      change_lr_when_available()

      -- shuffle data at epoch beginning
      --loaderTr:reset_batch_pointer()
    end

    -- do the optimization
    --require'mobdebug'.start()
    time = torch.tic()--------------------------------------
    --local _, lst = opt.optimFun(feval, params, opt.optimState)
    local _, lst = optim.rmsprop(feval, params, opt.optimState)
    local loss = lst[1]
    if opt.gpuId > 0 then cutorch.synchronize() end
    time = torch.toc(time)----------------------------------

    -- update
    lossesTr[i] = loss

    -- print?
    if i % opt.printFreq == 0 then
      print_progress()
    end

    -- evaluate (on validation set ?), update and save
    if i % opt.evalFreq == 0 then
      lossesVal[i], errVal[i] = eval_dataset(loaderVal)

      save_env{opt = opt,
        i = i, epoch = epoch,
        lossesTr = lossesTr, lossesVal = lossesVal, errVal = errVal,
        md = md, cri = cri,
        md_reset = md_reset,
      }
    end
    
    if i % 10 == 0 then collectgarbage() end
  end -- for i

  return env
end -- main

return this