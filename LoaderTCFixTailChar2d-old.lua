--- data loader for text classification. Fixed Character sequence at tail. organized as 2d data
-- (too slow, don't use this one)
-- data storage:
--   x: {N}, {M_i}, {W_j}
--   y: N
-- where:
--   N = #docs, M_i = #words per doc, W_j = #chars per word
-- guaranteed data batch:
--   xx: B, M, Q
--   yy: B
-- where:
--   B = batch size, M = #words, Q = #chars
--
require'torch'

--- class def
local LoaderTCFixTailChar2d = torch.class('LoaderTCFixTailChar2d')

function LoaderTCFixTailChar2d:__init(ffnData, batSize, seqLength, wordLength, arg)
	-- data
	self.ffnData = ffnData
	local data = torch.load(ffnData)

	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )

	self.batSize = batSize or error('no batSize')
	self.seqLength = seqLength or error('no seqLength')
	self.wordLength = wordLength or error('no wordLength')

	-- internal states
	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	self.isCuda = false

	self.charFill = arg.charFill or error('no arg.charFill')

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixTailChar2d:__tostring__()
	local str = torch.type(self) .. "\n"
	str = str .. 'created from ' .. self.ffnData .. "\n"
	str = str .. 'data size = ' .. #self.x .. "\n"
	str = str .. 'batSize = ' .. self.batSize .. "\n"
	str = str .. 'seqLength = ' .. self.seqLength .. "\n"
	str = str .. 'wordLength = ' .. self.wordLength .. "\n"
	str = str .. 'numBat = ' .. self.numBat .. "\n"
	str = str .. 'charFill = ' .. self.charFill .. "\n"
	return str
end

function LoaderTCFixTailChar2d:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixTailChar2d:cuda()
	self.isCuda = true
end

function LoaderTCFixTailChar2d:cpu()
	self.isCuda = false
end

function LoaderTCFixTailChar2d:next_batch()
	-- fetch the batch, X: size B x M x Q guaranteed Y: size B
	local xx = torch.ByteTensor(self.batSize, self.seqLength, self.wordLength):fill(self.charFill)
	local yy = torch.LongTensor(self.batSize)

	--require'mobdebug'.start()
	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		local ind = self.instIndex[ixBase + i]

		self:get_fix_char_seq2d(self.x[ind], xx[i], self.seqLength, self.wordLength)
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

function LoaderTCFixTailChar2d:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCFixTailChar2d:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end

function LoaderTCFixTailChar2d:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailChar2d:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCFixTailChar2d:get_fix_char_seq2d(src, dst, seqlen, wordlen)
	-- src: {seqlen}
	-- dst: seqlen, wordlen

	-- the source range at tail
	local srcBeg = math.max(#src-seqlen+1,1)
	local srcEnd = #src
	local realLenth = srcEnd-srcBeg+1

	-- the destination range at tail
	local dstBeg = seqlen-realLenth+1

	-- do the copying
	local j = dstBeg
	for i = srcBeg, srcEnd do
		local numChars = math.min(wordlen, src[i]:numel())
		dst[j][{ {1, numChars} }]:copy( src[i][{ {1, numChars} }] )

		j = j + 1
	end

end