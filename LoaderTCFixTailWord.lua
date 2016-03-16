--- word loader for text classification. Fixed length word token sequence at tail
-- expected input data
--   x: {B, [M_i]}
--   y: B
-- where B = #docs, M_i = #words in doc_i
-- guaranteed output data batch
--   xx: B, M
--   yy: B
--
require'torch'

--- class def
local LoaderTCFixTailWord = torch.class('LoaderTCFixTailWord')

function LoaderTCFixTailWord:__init(ffnData, batSize, seqLength, arg)
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
	self.seqLength = seqLength or 200

	-- internal states
	self.WORD_FILL = arg.wordFill or error('no arg.wordFill')

	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	self.isCuda = false

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixTailWord:__tostring__()
	local str = ""
	str = str .. torch.type(self) .. ":\n"
	str = str .. 'created from ' .. self.ffnData .. "\n"
	str = str .. 'data size = ' .. #self.x .. "\n"
	str = str .. 'batSize = ' .. self.batSize .. "\n"
	str = str .. 'seqLength = ' .. self.seqLength .. "\n"
	str = str .. 'numBat = ' .. self.numBat .. "\n"
	local tmp = {[false]='false', [true]='true'}
	str = str .. 'OrderRand = ' .. tmp[self.flagOrderRand] .. "\n"
	return str
end

function LoaderTCFixTailWord:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixTailWord:cuda()
	self.isCuda = true
end

function LoaderTCFixTailWord:cpu()
	self.isCuda = false
end

function LoaderTCFixTailWord:next_batch()
	-- fetch the batch, X: size B x M guaranteed Y: size B
	local xx = torch.Tensor():type(self.xtype):resize(self.batSize, self.seqLength):fill(self.WORD_FILL)
	local yy = torch.Tensor():type(self.ytype):resize(self.batSize)

	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]

		--require'mobdebug'.start()
		self:get_fix_word_seq(self.x[ind], xx[i], self.seqLength) 
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

function LoaderTCFixTailWord:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCFixTailWord:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailWord:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCFixTailWord:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end

function LoaderTCFixTailWord:get_fix_word_seq(src, dst, seqlen)
	-- src: n (variable length doc), tensor
	-- dst: M = seqlen, tensor

	-- the source range at tail
	local srcBeg = math.max(src:size(1)-seqlen+1,1)
	local srcEnd = src:size(1)
	local realLenth = srcEnd-srcBeg+1
	-- the destination range at tail
	local dstBeg = seqlen-realLenth+1
	local dstEnd = seqlen
	--
	dst[{ {dstBeg,dstEnd} }]:copy( src[{ {srcBeg,srcEnd} }] )

	return dst
end