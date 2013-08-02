-- The GainNoise module computes f(x) =  x * N(1, sigma^2)
-- where N the normal distribution. 


require 'torch'
require 'torch-env'
require 'nn'


local GainNoise, parent = torch.class('nn.GainNoise', 'nn.Module')

function GainNoise:__init(sigma)
    sigma = sigma or 0.001
    parent.__init(self)
    self.sigma = sigma
end

function GainNoise:updateOutput(input)
    self.output:typeAs(input):resizeAs(input):copy(input):mul(1 + self.sigma * torch.randn(1)[1])
    return self.output
end

function GainNoise:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput):mul(self.output[1]/input[1])
    return self.gradInput
end

