require'torch'
require'nn'
require'cudnn'
require'optim'
require'pl.path'
require'xlua'

require'OneHot'
require'util.misc'

local function setup_global()
  if opt.gpuId > 0 then
    require('cutorch').setDevice(opt.gpuId)
  end
  
  torch.manualSeed(opt.seed or 123)

end

local function get_dft_opt()
  local opt = {}
  
  opt.gpuId = 1
  opt.seed = 123

  opt.dataPath = 'data/ptb.lua'
  opt.envPath = 'path/to/saved/env'

  opt.batSize = 500

  return opt
end

local function main (opt)
  print('[options]')
  opt = opt or {}
  opt = tablex.merge(get_dft_opt(), opt)
  print(opt)
  setup_global()

  print('[training environment]')
  local env
  print('load from file ' .. opt.envPath)
  env = torch.load( opt.envPath )
  
  print('[data]')
  local _, _, loaderTe, vChar, vTag = dofile(opt.dataPath)

  print('[model]')
  local md, md_reset = env.md, env.md_reset
  local cri = env.cri or nn.ClassNLLCriterion()
  -- reset batch size and sequence length
  local newSeqLength, newBatSize = opt.seqLength, opt.batSize
  assert(newSeqLength==env.opt.seqLength,
    string.format("opt.segLength (=%d) must be equal to that of the saved model", newSeqLength)
  )
  md = md_reset(md, newSeqLength, newBatSize)
  print(md)
  
  print('[testing logic]')
  local function to_gpu()
    if opt.gpuId > 0 then -- to gpu
      require'cutorch'
      require'cunn'
      loaderTe:cuda(); -- lazy
      md:cuda(); cri:cuda();
    end
  end
  to_gpu()

  local function eval_dataset(loader)

    -- init
    loader:reset_batch_pointer()
    md:evaluate()
    local lossTotal = 0
    local numClasses = env.opt.numClasses
    local confWord = optim.ConfusionMatrix(numClasses) -- confusion matrix
    local outputsAll -- outputs over all data
    local targetsAll -- targets over all data

    print('evaluate on the dataset')
    local nb = loader:num_batches()
    print('#batches = ' .. nb)

    for ibat = 1, nb do
      -- data batch
      local inputs, targets = loader:next_batch()

      -- fprop
      local outputs = md:forward(inputs)
      local loss = cri:forward(outputs, targets)

      -- update loss
      lossTotal = lossTotal + loss

      -- update word level confusion
      confWord:batchAdd(outputs, targets)

      -- update outputsAll
      if not outputsAll then
        outputsAll = outputs:clone()
      else
        outputsAll = torch.cat(outputsAll, outputs, 1)
      end
      -- update targetsAll
      if not targetsAll then
        targetsAll = targets:clone()
      else
        targetsAll = torch.cat(targetsAll, targets, 1)
      end

      -- temp
      if ibat == 1 then
        print('during testing, output size = ' .. outputs:size(2))
      end

      xlua.progress(ibat, nb)
    end
    -- update
    local lossAvg = lossTotal/nb
    confWord:updateValids()

    md:training() -- restore
    return lossAvg, confWord, outputsAll, targetsAll
  end -- function

  local lossAvg, conf, outputsAll, targetsAll = eval_dataset(loaderTe)

  -- show info
  print('avg loss = ' .. lossAvg)
  print('per word accuracy = ' .. conf.totalValid)

  return outputsAll, targetsAll
end -- main

local outputsAll, targetsAll = main()
return outputsAll, targetsAll