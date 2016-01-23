--- loader for text classification. Fix word vector sequence at tail
-- xx: B, M, V
-- yy: B
require'torch'

--- helpers
local function get_fix_wordvec_seq(src, seqlen, vecsize)
	-- src: n x V
	-- dst: seqlen x V

	local dst = torch.zeros(seqlen, vecsize, 'torch.FloatTensor')

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
local LoaderTCFixTailWordvec = torch.class('LoaderTCFixTailWordvec')

function LoaderTCFixTailWordvec:__init(ffnData, batSize, seqLength)
	print('LoaderTCFixTailWordvec: creating from ' .. ffnData)
	local data = torch.load(ffnData)

	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )
	print('LoaderTCFixTailWordvec: data size = ' .. #self.x )
	self.vecSize = #self.x[1]:size(2)
	print('LoaderTCFixTailWordvec: vector size = ' .. self.vecSize)

	self.batSize = batSize or 500
	self.seqLength = seqLength or 200
	print('LoaderTCFixTailWordvec: batSize = ' .. self.batSize)
	print('LoaderTCFixTailWordvec: seqLength = ' .. self.seqLength)

	-- internal states
	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	print('LoaderTCFixTailWordvec: numBat = ' .. self.numBat)
	self.isCuda = false

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixTailWordvec:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixTailWordvec:cuda()
	self.isCuda = true
end

function LoaderTCFixTailWordvec:cpu()
	self.isCuda = false
end

function LoaderTCFixTailWordvec:training()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailWordvec:evaluate()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

function LoaderTCFixTailWordvec:next_batch()
	-- fetch the batch, X: size B x M x V guaranteed Y: size B
	local xx = torch.FloatTensor(self.batSize, self.seqLength, self.vecSize)
	local yy = torch.LongTensor(self.batSize)

	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]

		xx[i]:copy( get_fix_wordvec_seq(self.x[ind], self.seqLength) )
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

function LoaderTCFixTailWordvec:num_batches()
	return self.numBat -- truncate!
end

--- helper methods
function LoaderTCFixTailWordvec:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end