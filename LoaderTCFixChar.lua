--- char loader for text classification. Fix Character sequence
require'torch'

--- helpers

local function get_fix_char_seq(xchar, seqlen, char_null)
	assert(char_null, "char_null not defined")
	local xx = torch.ByteTensor(seqlen):fill(char_null)

	-- the source range
	local srcBeg = 1
	local srcEnd = math.min(srcBeg+seqlen-1, xchar:numel())
	-- the destination range
	local dstBeg = 1
	local dstEnd = dstBeg + (srcEnd-srcBeg)
	--
	xx[{ {dstBeg,dstEnd} }]:copy( xchar[{ {srcBeg,srcEnd} }] )

	return xx
end

--- class def
local LoaderTCFixChar = torch.class('LoaderTCFixChar')

function LoaderTCFixChar:__init(ffnData, batSize, seqLength, vocab)
	print('LoaderTCRandChar: creating from ' .. ffnData)
	local data = torch.load(ffnData)

	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )
	print('LoaderTCRandChar: data size = ' .. #self.x )

	self.batSize = batSize or 500
	self.seqLength = seqLength or 200
	print('LoaderTCRandChar: batSize = ' .. self.batSize)
	print('LoaderTCRandChar: seqLength = ' .. self.seqLength)

	-- internal states
	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	print('LoaderTCRandChar: numBat = ' .. self.numBat)
	self.isCuda = false

	print('LoaderTCRandChar: #vocab = ' .. tablex.size(vocab))
	self.char_null = tablex.size(vocab) + 1
	print('LoaderTCRandChar: char null = ' .. self.char_null)

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixChar:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixChar:cuda()
	self.isCuda = true
end

function LoaderTCFixChar:cpu()
	self.isCuda = false
end

function LoaderTCFixChar:training()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixChar:evaluate()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

function LoaderTCFixChar:next_batch()
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

function LoaderTCFixChar:num_batches()
	return self.numBat -- truncate!
end

--- helper methods
function LoaderTCFixChar:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end