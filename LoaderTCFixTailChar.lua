--- loader for text classification. Fix Character sequence at tail
require'torch'

--- helpers
local function get_fix_char_seq(src, seqlen, char_null)
	assert(char_null, "char_null not defined")
	local dst = torch.ByteTensor(seqlen):fill(char_null)

	-- the source range at tail
	local srcBeg = math.max(src:numel()-seqlen+1,1)
	local srcEnd = src:numel()
	local realLenth = srcEnd-srcBeg+1
	-- the destination range at tail
	local dstBeg = seqlen-realLenth+1
	local dstEnd = seqlen
	--
	dst[{ {dstBeg,dstEnd} }]:copy( src[{ {srcBeg,srcEnd} }] )

	return dst
end

--- class def
local LoaderTCFixTailChar = torch.class('LoaderTCFixTailChar')

function LoaderTCFixTailChar:__init(ffnData, batSize, seqLength, vocab)
	print('LoaderTCFixTailChar: creating from ' .. ffnData)
	local data = torch.load(ffnData)

	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )
	print('LoaderTCFixTailChar: data size = ' .. #self.x )

	self.batSize = batSize or 500
	self.seqLength = seqLength or 200
	print('LoaderTCFixTailChar: batSize = ' .. self.batSize)
	print('LoaderTCFixTailChar: seqLength = ' .. self.seqLength)

	-- internal states
	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	print('LoaderTCFixTailChar: numBat = ' .. self.numBat)
	self.isCuda = false

	print('LoaderTCFixTailChar: #vocab = ' .. tablex.size(vocab))
	self.char_null = tablex.size(vocab) + 1
	print('LoaderTCFixTailChar: char null = ' .. self.char_null)

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
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

function LoaderTCFixTailChar:training()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailChar:evaluate()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

function LoaderTCFixTailChar:next_batch()
	-- fetch the batch, X: size B x M guaranteed Y: size B
	local xx = torch.ByteTensor(self.batSize, self.seqLength)
	local yy = torch.LongTensor(self.batSize)


	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]

		xx[i]:copy( get_fix_char_seq(self.x[ind], self.seqLength, self.char_null) )
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

--- helper methods
function LoaderTCFixTailChar:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end