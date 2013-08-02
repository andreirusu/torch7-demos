-- The GainNoise module computes f(x) =  x * N(1, sigma^2)
-- where N the normal distribution. 


require 'torch'
require 'torch-env'
require 'nn'


function randp(lambda)
    u = torch.rand(1000)
    p = 1
    k = 0
    L = math.exp(-lambda)
    repeat
        k = k + 1
        p = p * u[k]
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

