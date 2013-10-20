--- DISTORT LAYER
--- PURPOSE: Enhance the dataset by adding a random transformation from the set {Identity, Mirror, VerticalFlip, Jitter} to the input.

require 'image'
require 'nn'
require 'nnd'

local Distort, Parent = torch.class('nn.Distort', 'nn.Module')

function Distort:__init()
	Parent.__init(self)
	self.enabled = true
end


function Distort:enable()
	self.enabled = true
end


function Distort:disable()
	self.enabled = false
end


function Distort:updateOutput(input)
	self.output:resizeAs(input):copy(input)
	if self.enabled then
		-- flip a fair coin and apply hflip
		if torch.rand(1):mul(2):ceil()[1] == 1 then
			image.hflip(self.output, input)
        end
		
        -- flip a fair coin and apply jitter
		if torch.rand(1):mul(2):ceil()[1] == 1 then
            -- jitter by a random number of pixels (between 1 and 3) in either direction 
            local width, height
            if self.output:dim() == 2 then
                width = self.output:size(1)
                height = self.output:size(2)
            elseif self.output:dim() == 3 then
                width = self.output:size(2)
                height = self.output:size(3)
            else
                error('Distort layer does not support 4D input. Use network spec file to disable distorsions if 4D input is required.')
            end
            local w_pixels = torch.rand(1):mul(3):ceil()[1] 
            local w_direction = torch.rand(1):mul(2):ceil()[1] - 1  
            local h_pixels = torch.rand(1):mul(3):ceil()[1] 
            local h_direction = torch.rand(1):mul(2):ceil()[1] - 1  
            
            self.output = image.crop(self.output, w_direction * w_pixels , 
                                                 h_direction * h_pixels, 
                                                 width - (1-w_direction) * w_pixels,
                                                 height - (1-h_direction) * h_pixels)
            self.output = image.scale(self.output, width, height, 'bilinear')
		end
	end
	return self.output
end


function Distort:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	return self.gradInput
end


function Distort:__tostring__()
	local function mode() if self.enabled then return 'on' else return 'off' end end
	return torch.typename(self) .. ' [' .. mode() .. ']'
end
