--- char loader for text classification. Fixed length word sequence at tail. each word is a char window at word center.
-- expected input data batch:
--   x: {B, [N_i]}
--   y: B
-- where N_i = #chars in doc_i
-- guaranteed output data batch:
--   xx: B, M, Q (,V)
--   yy: B
-- where M = #words, Q = window size = #chars in window, V = vocabulary size
--
require'torch'

--- class def
local LoaderTCFixTailCharWordwin = torch.class('LoaderTCFixTailCharWordwin')

function LoaderTCFixTailCharWordwin:__init(ffnData, batSize, seqLength, winSize, arg)
	-- data
	self.ffnData = ffnData
	local data = torch.load(ffnData)
	self.x = data.x
	self.y = data.y
	assert( #self.x == self.y:numel() )

	-- internal states
	self.charFill = arg.charFill or error('no arg.charFill')

	--require'mobdebug'.start()
	-- data layout
	self.batSize = batSize or error('no batSize')
	self.seqLength = seqLength or error('no seqLength')
	self.winSize = winSize or error('no winSize')
	self:make_ptr_to_word_all()

	self.iBat = 0
	self.numBat = math.floor(#self.x/self.batSize) -- truncate!
	self.isCuda = false

	self.flagOrderRand = true
	self.instIndex = torch.randperm(#self.x):long()

	collectgarbage()
end

function LoaderTCFixTailCharWordwin:__tostring__()
	local str = ""
	str = str .. torch.type(self) .. ":\n"
	str = str .. 'created from ' .. self.ffnData .. "\n"
	str = str .. 'data size = ' .. #self.x .. "\n"
	str = str .. 'batSize = ' .. self.batSize .. "\n"
	str = str .. 'seqLength = ' .. self.seqLength .. "\n"
	str = str .. 'winSize = ' .. self.winSize .. "\n"
	str = str .. 'numBat = ' .. self.numBat .. "\n"
	local tmp = {[false]='false', [true]='true'}
	str = str .. 'OrderRand = ' .. tmp[self.flagOrderRand] .. "\n"
	return str
end

function LoaderTCFixTailCharWordwin:reset_batch_pointer()
	self.iBat = 0
	self:reset_order()
end

function LoaderTCFixTailCharWordwin:cuda()
	self.isCuda = true
end

function LoaderTCFixTailCharWordwin:cpu()
	self.isCuda = false
end

function LoaderTCFixTailCharWordwin:next_batch()
	-- fetch the batch, X: B x M x Q  Y: B
	local xx = torch.LongTensor(self.batSize, self.seqLength, self.winSize):fill(self.charFill)
	local yy = torch.LongTensor(self.batSize)

	local ixBase = self.iBat * self.batSize
	for i = 1, self.batSize do
		--require'mobdebug'.start()

		local iOrig = self.instIndex[ixBase + i]

		self:fill_fixlen_char_wordwin(
			self.x[iOrig], self.ptrWords[iOrig],
			xx[i], self.numSent, self.seqLength
		)
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

function LoaderTCFixTailCharWordwin:num_batches()
	return self.numBat -- truncate!
end

--- other methods
function LoaderTCFixTailCharWordwin:set_order_rand()
	self.flagOrderRand = true
	self:reset_batch_pointer()
end

function LoaderTCFixTailCharWordwin:set_order_natural()
	self.flagOrderRand = false
	self:reset_batch_pointer()
end

--- helper methods
function LoaderTCFixTailCharWordwin:reset_order()
	if self.flagOrderRand == true then
		self.instIndex = torch.randperm(#self.x):long()
	else
		self.instIndex = torch.range(1, #self.x):long()
	end
end

function LoaderTCFixTailCharWordwin:fill_fixlen_char_wordwin(src, srcPtrWord, dst, M, Q)
	-- src: [n] tensor, variable length chars in doc
	-- srcPtrWord: {K} table. ptr to word center. K = #words
	-- dst: [M, Q] tensor

	local n = src:numel() -- #chars in doc
	local K = #srcPtrWord -- #words in doc
	local realNumWord = math.min(M, K)

	for i = 1, realNumWord do
		-- word count: src and dst
		local iSrc = K - i + 1
		local iDst = M - i + 1

		-- char seq: src and dst
		local LeftLen = math.floor( (Q + 1) / 2 )
		local RightLen = math.ceil( (Q + 1) / 2 )

		local jSrcCen = srcPtrWord[iSrc]
		local jSrcBeg = math.max(jSrcCen - LeftLen + 1, 1)
		local jSrcEnd = math.min(jSrcCen + RightLen - 1, n)

		local RealLeftLen = jSrcCen - jSrcBeg + 1
		local RealRightLen = jSrcEnd - jSrcCen + 1

		local jDstCen = LeftLen
		local jDstBeg = jDstCen - RealLeftLen + 1
		local jDstEnd = jDstCen + RealRightLen - 1

		-- do the copying
		dst[iDst][{ {jDstBeg, jDstEnd} }]:copy( src[{ {jSrcBeg, jSrcEnd} }] )
	end
end

function LoaderTCFixTailCharWordwin:make_ptr_to_word_doc(xx)
	-- xx: [n] tensor, chars in current doc
	local ptrWords = {} -- {M} word center for current doc

	-- closure, will update the iSentBeg, iSentEnd when called
	local iWordBeg, iWordEnd = 0, 0
	local function next_word_position()
		local flagInsideNewWord = false
		local i = iWordEnd + 1

		while true do
			-- if pass-end?
			if i > xx:numel() then
				iWordBeg, iWordEnd = nil, nil
				break
			end

			-- check current char
			local c = xx[i]
			if c == self.charWordDlm then -- found a word delimiter
				if flagInsideNewWord == true then -- at new word pass-end: roll back, quit
					iWordEnd = i - 1
					break
				end
			else -- found a non-delimiter
				if  flagInsideNewWord == false then -- at new word beginning
					iWordBeg = i
					flagInsideNewWord = true
				end
			end

			-- advance to next char
			i = i + 1
		end -- while
	end

	-- find all sentences
	while true do
		next_word_position()
		if iWordBeg and iWordEnd then
			local iWordCen = math.floor( (iWordBeg + iWordEnd) / 2 )
			table.insert(ptrWords, iWordCen)
		else
			break
		end
	end

	return ptrWords
end

function LoaderTCFixTailCharWordwin:make_ptr_to_word_all()
	-- set self.ptrWords: {B, {S_n}}
	self.ptrWords = {}

	-- for each doc
	for i = 1, #self.x do
		local ptrWordCur = self:make_ptr_to_word_doc(self.x[i])
		table.insert(self.ptrWords, ptrWordCur)
	end
end