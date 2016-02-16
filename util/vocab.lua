--- vocabulary utility functions

local this = {}

this.str2tensor = function(str, vocab)
    error('not implemented.')
    --    local ten = torch.Tensor(1, #str)
    --    for i = 1, #str do
    --        local curChar = string.sub(str,i,i)
    --        ten[1][i] = vocab[curChar]
    --    end
    --    ten = ten:cuda()
    --    return ten
end

this.tensor2str = function(ten, ivocab, nullfill)
    local str = ""
    nullfill = nullfill or "_"
    for i = 1, ten:numel() do
        local char = ivocab[ten[i]] or nullfill
        str = str .. char
    end
    return str
end

this.make_inverse_vocabulary = function(vocab)
    local ivocab = {}
    for c,i in pairs(vocab) do ivocab[i] = c end
    return ivocab
end

return this

