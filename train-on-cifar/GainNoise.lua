-- The GainNoise module computes f(x) =  x * N(1, sigma^2)
-- where N the normal distribution. 


require 'torch'
require 'torch-env'
require 'nn'


function randp(lambda)
    local bsize = 10000
    local u 
    local p = 1
    local k = 0
    local L = math.exp(-lambda)
    repeat
        if k % bsize == 0 then
            u = torch.rand(bsize)
        end
        p = p * u[1 + k % bsize ]
        k = k + 1
    until p < L 
    return (k-1)
end


local GainNoise, parent = torch.class('nn.GainNoise', 'nn.Module')

function GainNoise:__init(sigma)
    sigma = sigma or 0
    parent.__init(self)
    self.sigma = sigma
    self.enabled = (sigma > 0)
end

function GainNoise:setEnabled(val)
    self.enabled = val
end

function GainNoise:setSigma(val)
    self.sigma = val
end

function GainNoise:updateOutput(input)
    self.scaler = self.sigma 
    if self.enabled then
        --self.scaler = self.scaler + self.sigma * torch.randn(1)[1]
        self.scaler = randp(self.sigma) 
    end
    self.output:typeAs(input):resizeAs(input):copy(input):mul(self.scaler)
    return self.output
end

function GainNoise:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput):mul(self.scaler)
    return self.gradInput
end

