--- data loader for text classification. Fixed Character sequence at tail
-- guaranteed data batch:
-- xx: B, M
-- yy: B
--
require'torch'

--- class def
local LoaderTCFixTailChar = torch.class('LoaderTCFixTailChar')

function LoaderTCFixTailChar:__init(ffnData, batSize, seqLength, arg)
	-- data
	self.ffnData = ffnData
	local data = torch.load(ffnData)

	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )

	self.batSize = batSize or 500
	self.seqLength = seqLength or 200

	-- internal states
	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	self.isCuda = false

	self.CHAR_FILL = arg.charFill or error('no arg.charFill')

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixTailChar:__tostring__()
	local str = torch.type(self) .. "\n"
	str = str .. 'created from ' .. self.ffnData .. "\n"
	str = str .. 'data size = ' .. #self.x .. "\n"
	str = str .. 'batSize = ' .. self.batSize .. "\n"
	str = str .. 'seqLength = ' .. self.seqLength .. "\n"
	str = str .. 'numBat = ' .. self.numBat .. "\n"
	str = str .. 'charFill = ' .. self.charFill .. "\n"
	return str
end

function LoaderTCFixTailChar:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixTailChar:cuda()
	self.isCuda = true
end

function LoaderTCFixTailChar:cpu()
	self.isCuda = false
end

function LoaderTCFixTailChar:next_batch()
	-- fetch the batch, X: size B x M guaranteed Y: size B
	local xx = torch.ByteTensor(self.batSize, self.seqLength):fill(self.CHAR_FILL)
	local yy = torch.LongTensor(self.batSize)

	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]

		self:get_fix_char_seq(self.x[ind], xx[i], self.seqLength)
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

function LoaderTCFixTailChar:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCFixTailChar:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end

function LoaderTCFixTailChar:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailChar:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCFixTailChar:get_fix_char_seq(src, dst, seqlen)
	-- src: seqlen, ByteTensor
	-- dst: seqlen, ByteTensor

	-- the source range at tail
	local srcBeg = math.max(src:numel()-seqlen+1,1)
	local srcEnd = src:numel()
	local realLenth = srcEnd-srcBeg+1

	-- the destination range at tail
	local dstBeg = seqlen-realLenth+1
	local dstEnd = seqlen

	-- do the copying
	dst[{ {dstBeg,dstEnd} }]:copy( src[{ {srcBeg,srcEnd} }] )
end