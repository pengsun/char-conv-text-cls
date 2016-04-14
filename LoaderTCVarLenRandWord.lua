--- word loader for text classification. Variable length, but fixed length word token sequence for each batch.
-- expected input data
--   x: {B, [M_i]}
--   y: B
-- where B = #docs, M_i = #words in doc_i
-- guaranteed output data batch
--   xx: B, MM_j
--   yy: B
-- where MM_j = max doc length of j-th batch
--
require'torch'

--- class def
local LoaderTCVarLenRandWord = torch.class('LoaderTCVarLenRandWord')

function LoaderTCVarLenRandWord:__init(ffnData, batSize, arg)
	-- data
	self.ffnData = ffnData
	local data = torch.load(ffnData)
	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )
	self.xtype = self.x[1]:type()
	self.ytype = self.y:type()

	-- data layout
	self.batSize = batSize or 500

	-- internal states
	self.WORD_FILL = arg.wordFill or error('no arg.wordFill')

	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	self.isCuda = false

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCVarLenRandWord:__tostring__()
	local str = ""
	str = str .. torch.type(self) .. ":\n"
	str = str .. 'created from ' .. self.ffnData .. "\n"
	str = str .. 'data size = ' .. #self.x .. "\n"
	str = str .. 'batSize = ' .. self.batSize .. "\n"
	str = str .. 'numBat = ' .. self.numBat .. "\n"
	local tmp = {[false]='false', [true]='true'}
	str = str .. 'OrderRand = ' .. tmp[self.flagOrderRand] .. "\n"
	return str
end

function LoaderTCVarLenRandWord:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCVarLenRandWord:cuda()
	self.isCuda = true
end

function LoaderTCVarLenRandWord:cpu()
	self.isCuda = false
end

function LoaderTCVarLenRandWord:next_batch()
	local ixBase = self.iBat * self.batSize

	-- determine batch length: MM, the longest doc (word seq)
	local MM = -1
	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]
		local curLen = self.x[ind]:numel()
		MM = (MM < curLen) and curLen or MM
	end

	-- fetch the batch, X: size B x MM guaranteed Y: size B
	local xx = torch.Tensor():type(self.xtype):resize(self.batSize, MM):fill(self.WORD_FILL)
	local yy = torch.Tensor():type(self.ytype):resize(self.batSize)

	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]

		--require'mobdebug'.start()
		self:get_fix_word_seq(self.x[ind], xx[i])
		yy[i] = self.y[ind]
	end

	-- advance batch pointer
	self.iBat = self.iBat + 1
	if self.iBat >= self:num_batches() then
		self:reset_batch_pointer()
	end

	-- to cuda?
	if self.isCuda then
		xx = xx:cuda(); yy = yy:cuda();
	end

	return xx, yy
end

function LoaderTCVarLenRandWord:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCVarLenRandWord:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCVarLenRandWord:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCVarLenRandWord:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end

function LoaderTCVarLenRandWord:get_fix_word_seq(src, dst)
	-- src: n (variable length doc), tensor
	-- dst: MM, tensor

	-- src <= dst
	assert(src:numel() <= dst:numel())
	-- the destination range at head
	local iBeg = 1
	local iEnd = src:numel()
	--
	dst[{ {iBeg, iEnd} }]:copy(src)
end