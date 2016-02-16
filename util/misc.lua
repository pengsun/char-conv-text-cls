--- miscellaneous utility functions
local this = {}

--- path
this.ensure_path = function (p)
    if not path.isdir(p) then
        print('creating path ' .. p)
        assert(path.mkdir(p), 'creation failed')
    end
end

--- model
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
--- Resize the output, gradInput, etc temporary tensors to zero (so that the
-- on disk size is smaller)
-- code by jonathantompson, from https://github.com/torch/nn/issues/112
this.cleanup_model = cleanup_model

--- printing
this.print_tensor_asrow = function(ten)
    local t = ten:reshape(ten:numel())
    for i = 1, t:numel() do
        io.write( t[i] .. ', ')
    end
    io.write("\n")
    io.flush()
end

return this