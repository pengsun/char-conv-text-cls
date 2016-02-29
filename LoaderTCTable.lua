--- Aggregate other loaders as table output
require'torch'

--- class def
local LoaderTCTable = torch.class('LoaderTCTable')

function LoaderTCTable:__init(loaderOne, loaderTwo)
    self.loaderOne = loaderOne
    self.loaderTwo = loaderTwo
    assert(loaderOne:num_batches() == loaderTwo:num_batches())

    self:enforce_order()
    self.flagCuda = false
end

function LoaderTCTable:__tostring__()
    local str = torch.type(self) .. "\n"
    str = str .. '------------------\n'
    str = str .. self.loaderOne:__tostring__()
    str = str .. '------------------\n'
    str = str .. self.loaderTwo:__tostring__()
    return str
end

function LoaderTCTable:reset_batch_pointer()
    self.loaderOne:reset_batch_pointer()
    self.loaderTwo:reset_batch_pointer()

    self:enforce_order()
end

function LoaderTCTable:cuda()
    self.loaderOne:cuda()
    self.loaderTwo:cuda()

    self.flagCuda = true
end

function LoaderTCTable:cpu()
    self.loaderOne:cpu()
    self.loaderTwo:cpu()

    self.flagCuda = false
end

function LoaderTCTable:next_batch()
    -- fetch the batch, X: {B x size1, B x size2} guaranteed, Y: size B
    --require'mobdebug'.start()
    local xx1, yy1 = self.loaderOne:next_batch() -- B x size1
    local xx2, yy2 = self.loaderTwo:next_batch() -- B X size2

    -- xx
    local xx = {}
    xx[1] = xx1:clone()
    xx[2] = xx2:clone()

    -- yy
    local yy = yy1:clone()

    return xx, yy
end

function LoaderTCTable:num_batches()
    return self.loaderOne:num_batches()
end

-- other medthods
function LoaderTCTable:set_order_rand()
    self.loaderOne:set_order_rand()
    self.loaderTwo:set_order_rand()

    self:enforce_order()
end

function LoaderTCTable:set_order_natural()
    self.loaderOne:set_order_natural()
    self.loaderTwo:set_order_natural()

    self:enforce_order()
end

function LoaderTCTable:reset_order()
    self.loaderOne:reset_order()
    self.loaderTwo:reset_order()

    self:enforce_order()
end

function LoaderTCTable:enforce_order()
    self.loaderTwo.instIndex = self.loaderOne.instIndex:clone()
end