--- miscellaneous utility functions
local this = {}

-- Resize the output, gradInput, etc temporary tensors to zero (so that the
-- on disk size is smaller)
-- code by jonathantompson, from https://github.com/torch/nn/issues/112
local function cleanup_model (node)

    local function zeroDataSize(data)
        if type(data) == 'table' then
            for i = 1, #data do
                data[i] = zeroDataSize(data[i])
            end
        elseif type(data) == 'userdata' then
            data = torch.Tensor():typeAs(data)
        end
        return data
    end

    if node.output ~= nil then
        node.output = zeroDataSize(node.output)
    end
    if node.gradInput ~= nil then
        node.gradInput = zeroDataSize(node.gradInput)
    end
    if node.finput ~= nil then
        node.finput = zeroDataSize(node.finput)
    end
    -- Recurse on nodes with 'modules'
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i = 1, #node.modules do
                local child = node.modules[i]
                cleanup_model(child)
            end
        end
    end

    collectgarbage()
end
this.cleanup_model = cleanup_model

-- util global functions
this.print_tensor_asrow = function(ten)
    local t = ten:reshape(ten:numel())
    for i = 1, t:numel() do
        io.write( t[i] .. ', ')
    end
    io.write("\n")
    io.flush()
end

-- vocab
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