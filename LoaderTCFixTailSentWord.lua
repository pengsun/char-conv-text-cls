--- word loader for text classification. Fixed length sentence x word token sequence at tail
-- xx: B, S, M (,V)
-- yy: B
require'torch'

--- class def
local LoaderTCFixTailSentWord = torch.class('LoaderTCFixTailSentWord')

function LoaderTCFixTailSentWord:__init(ffnData, batSize, numSent, seqLength, arg)
	-- data
	self.ffnData = ffnData
	local data = torch.load(ffnData)
	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )

	-- internal states
	self.wordFill = arg.wordFill or error('no arg.wordFill')
	self.sentSym1 = arg.sentSym1 or error('no arg.sentSym1')
	self.sentSym2 = arg.sentSym2 or error('no arg.sentSym2')
	self.sentSym3 = arg.sentSym3 or error('no arg.sentSym3')
	self.sentSym4 = arg.sentSym4 or error('no arg.sentSym4')

	--require'mobdebug'.start()
	-- data layout
	self.batSize = batSize or error('no batSize')
	self.numSent = numSent or error('no numSent')
	self.seqLength = seqLength or error('no seqLength')
	self:make_ptr_to_sent_all()

	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	self.isCuda = false

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixTailSentWord:__tostring__()
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

function LoaderTCFixTailSentWord:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixTailSentWord:cuda()
	self.isCuda = true
end

function LoaderTCFixTailSentWord:cpu()
	self.isCuda = false
end

function LoaderTCFixTailSentWord:next_batch()
	-- fetch the batch, X: B x S x M  Y: B
	local xx = torch.LongTensor(self.batSize, self.numSent, self.seqLength):fill(self.wordFill)
	local yy = torch.LongTensor(self.batSize)

	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		--require'mobdebug'.start()

		local iOrig = self.instIndex[ixBase + i]

		self:get_fix_sent_word_seq(self.x[iOrig], self.ptrSents[iOrig], xx[i], self.numSent, self.seqLength)
		yy[i] = self.y[iOrig]
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

function LoaderTCFixTailSentWord:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCFixTailSentWord:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailSentWord:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCFixTailSentWord:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end

function LoaderTCFixTailSentWord:get_fix_word_seq(src, dst, seqlen)
	-- src: [n] tensor, variable length sent
	-- dst: [seqlen] tensor

	-- the source range at tail
	local srcBeg = math.max(src:size(1)-seqlen+1,1)
	local srcEnd = src:size(1)
	local realLenth = srcEnd-srcBeg+1

	-- the destination range at tail
	local dstBeg = seqlen-realLenth+1
	local dstEnd = seqlen

	-- do the real copying
	dst[{ {dstBeg,dstEnd} }]:copy( src[{ {srcBeg,srcEnd} }] )
end

function LoaderTCFixTailSentWord:get_fix_sent_word_seq(src, srcPtrSent, dst, S, M)
	-- src: [n] tensor, variable length doc
	-- srcPtrSent: {K, {2}} table. ptr to sentence beg, end
	-- dst: [S, M] tensor
	local K = #srcPtrSent
	local realNumSent = math.min(S, K)

	for i = 1, realNumSent do
		-- src sentence
		local iSrc = K - i + 1
		local ptrBeg, ptrEnd = srcPtrSent[iSrc][1], srcPtrSent[iSrc][2]
		local sentSrc = src[{ { ptrBeg, ptrEnd } }]

		-- dst sentence
		local iDst = S - i + 1
		local sentDst = dst[iDst]

		-- copy the row, i.e., the sentence
		self:get_fix_word_seq(sentSrc, sentDst, M)
	end
end

function LoaderTCFixTailSentWord:make_ptr_to_sent_doc(xx)
	-- xx: [n] tensor, words in current doc
	local ptrSents = {} -- {S, {2}} for current doc

	-- closure, will update the iSentBeg, iSentEnd when called
	local iSentBeg, iSentEnd = 0, 0
	local function next_sent_position()
		iSentBeg = iSentEnd + 1

		local flagSentTail = false
		local i = iSentEnd + 1

		while true do
			-- if pass-end?
			if i > xx:numel() then
				iSentBeg, iSentEnd = nil, nil
				break
			end

			-- check current word
			local word = xx[i]
			if flagSentTail == false then -- previous word inside a sentence
				if word == self.sentSym1 or word == self.sentSym2 or word == self.sentSym3 or word == self.sentSym4 then
					-- found a sentence tail symbol
					flagSentTail = true
				end
			else -- previous word inside sentence-tail symbols
				if word ~= self.sentSym1 and word ~= self.sentSym2 and word ~= self.sentSym3 and word ~= self.sentSym4 then
					-- found a non-sentence tail symbol, beginning of new sent
					iSentEnd = i - 1
					break
				end
			end

			i = i + 1
		end -- while
	end

	-- find all sentences
	while true do
		next_sent_position()
		if iSentBeg and iSentEnd then
			table.insert(ptrSents, {iSentBeg, iSentEnd})
		else
			break
		end
	end

	return ptrSents
end

function LoaderTCFixTailSentWord:make_ptr_to_sent_all()
	-- set self.ptrSents: {B, {S_n}}
	self.ptrSents = {}

	-- for each doc
	for i = 1, #self.x do
		local ptrSentCur = self:make_ptr_to_sent_doc(self.x[i])
		table.insert(self.ptrSents, ptrSentCur)
	end
end