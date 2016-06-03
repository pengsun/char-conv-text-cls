require'nn'
require'nngraph'

B = 7
C = 3
H = 5
kS = 3
pad = 1

-- input
input = torch.Tensor(B, C, 10, 10)

-- model
x = nn.Identity() ()

x1d = nn.SpatialConvolution(C, H, kS,kS, 1,1, pad,pad) (x)
x1 = nn.ReLU(true)(x1d)

x2d = nn.SpatialConvolution(H, H, kS,kS, 1,1, pad,pad) (x1)
x2 = nn.ReLU(true)(x2d)

x3d = nn.SpatialConvolution(H, H, kS,kS, 1,1, pad,pad) (x2)
x3 = nn.ReLU(true)(x3d)

m = nn.gModule({x}, {x1, x2, x3})

-- parameter accees
pT, gradPT = m:parameters()
p, gradP = m:getParameters()

-- fprop
y = m:forward(input)

-- bprop
grady = {}
for _, t in ipairs(y) do
    table.insert(grady, torch.Tensor():resizeAs(t):fill(1))
end
gradInput = m:backward(input, grady)