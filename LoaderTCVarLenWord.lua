--- word loader for text classification. Variable length word token sequence. Grouped by length
-- expected input data
--   x: {B, [M_i]}
--   y: B
-- where B = #docs, M_i = #words in doc_i
-- guaranteed output data batch
--   xx: B, MM_j
--   yy: B
-- where MM_j = max doc length of j-th group
--
require'torch'

--- class def
local LoaderTCVarLenWord = torch.class('LoaderTCVarLenWord')

function LoaderTCVarLenWord:__init(ffnData, batSize, arg)
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

	self.numBat = math.floor(#self.x/self.batSize) --  truncate without warninging!
	self.flagOrderRand = true

	self:reset_batch_pointer()
	self:group_batch()

	self.isCuda = false

	collectgarbage()
end

function LoaderTCVarLenWord:__tostring__()
	local str = ""
	str = str .. torch.type(self) .. ":\n"
	str = str .. 'created from ' .. self.ffnData .. "\n"

	local num = #self.sortxlen
	str = str .. 'min len = ' .. self.sortxlen[1] .. "\n"
	str = str .. 'max len = ' .. self.sortxlen[num] .. "\n"
	local indMed = math.ceil(#self.sortxlen/2)
	str = str .. 'median len = ' .. self.sortxlen[indMed] .. "\n"
	local avg = tablex.reduce('+',self.sortxlen)/num
	str = str .. 'mean len = ' .. math.ceil(avg) .. "\n"

	str = str .. 'data size = ' .. #self.x .. "\n"
	str = str .. 'batSize = ' .. self.batSize .. "\n"
	str = str .. 'numBat = ' .. self.numBat .. "\n"
	local tmp = {[false]='false', [true]='true'}
	str = str .. 'OrderRand = ' .. tmp[self.flagOrderRand] .. "\n"
	return str
end

function LoaderTCVarLenWord:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCVarLenWord:cuda()
	self.isCuda = true
end

function LoaderTCVarLenWord:cpu()
	self.isCuda = false
end

function LoaderTCVarLenWord:next_batch()
	-- determine batch length: MM
	local iNexBat = self.batIndex[self.iBat + 1]
	local instMax = self.batInstIndex[iNexBat][-1] -- last one has the maximum length
	local MM = self.xlen[instMax]

	-- fetch the batch, X: size B x MM guaranteed Y: size B
	local xx = torch.Tensor():type(self.xtype):resize(self.batSize, MM):fill(self.WORD_FILL)
	local yy = torch.Tensor():type(self.ytype):resize(self.batSize)

	for i = 1, self.batSize do
		local ind = self.batInstIndex[iNexBat][i]

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

function LoaderTCVarLenWord:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCVarLenWord:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCVarLenWord:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCVarLenWord:group_batch()
	-- sort doc by the length
	self.xlen = {}
	for _, xx in pairs(self.x) do
		table.insert(self.xlen, xx:numel())
	end
	local sortInstIndex = {}
	self.sortxlen = {}
	for ind, length in tablex.sortv(self.xlen) do
		table.insert(sortInstIndex, ind)
		table.insert(self.sortxlen, length)
	end

	-- make batch-instance matrix by reshaping
	assert(self.numBat*self.batSize == #sortInstIndex, "incorrect batch size: not divisible")
	self.batInstIndex = torch.Tensor(sortInstIndex):long()
		:resize(self.numBat, self.batSize)
end

function LoaderTCVarLenWord:reset_order()
	if self.flagOrderRand == true then
		self.batIndex = torch.randperm(self.numBat):long()
	else
		self.batIndex = torch.range(1, self.numBat):long()
	end
end

function LoaderTCVarLenWord:get_fix_word_seq(src, dst)
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