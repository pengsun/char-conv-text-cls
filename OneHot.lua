--[[ ]]--
-- convert inputs to One Hot vectors with dimension V (= vocabulary size)
-- inputs: sz_1, ..., sz_M
-- outputs: sz_1, ..., sz_M, V
require'nn'

local OneHot, parent = torch.class('OneHot', 'nn.Module')


function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self._eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input)
  assert(self._eye, 'self._eye not properly set!')

  -- initialize self.output to proper size
  local sz = input:size()
  sz:resize(input:dim() + 1)
  sz[input:dim() + 1] = self.outputSize
  self.output:resize(sz):zero()

  -- fill the one hot representation tensor
  self.output:copy(
    self._eye:index(1, input:reshape(input:numel()))
  )
  return self.output
end

