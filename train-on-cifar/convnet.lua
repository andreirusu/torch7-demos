require "nn"
require "nnd"
require "nndx"
require "image"

function getModel(opt)
    local model 
    if opt.load == '' then
       -- define model to train
        model = {
            opt = opt,
            net = nn.Sequential(),
            criterion = nn.ClassNLLCriterion()
        }
        local net = model.net
        local classes = getClassLabels()
        ------------------------------------------------------------
        -- convolutional network
        ------------------------------------------------------------
        -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
        net:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
        net:add(nnd.Rectifier())
        net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        net:add(nn.SpatialContrastiveNormalization(16, image.gaussian1D(7)))
        -- stage 2 : filter bank -> squashing -> max pooling
        net:add(nn.SpatialConvolutionMap(nn.tables.random(16, 32, 4), 5, 5))
        net:add(nnd.Rectifier())
        net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 
        net:add(nn.SpatialContrastiveNormalization(32, image.gaussian1D(5)))
        -- stage 3 : standard 2-layer neural network
        net:add(nn.Reshape(32*5*5))
        net:add(nn.Linear(32*5*5, 32))
        net:add(nnd.Rectifier())
        net:add(nn.Linear(32,#classes))
        net:add(nn.LogSoftMax())
        ------------------------------------------------------------
    else
        print('<convnet> reloading previously trained network: '..opt.load)
        model = torch.load(opt.load)
        -- set new all new options if not in test mode
        model.opt = opt
    end
    print('<convnet> model:')
    print(model) 
    return model
end


function displayModel(model, input)
    local opt = model.opt
   iter = iter or 0
   require 'image'
   if iter%100 == 0 then
      if opt.model == 'convnet' then
         win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
         win_w1 = image.display{image=model.net:get(1).weight, zoom=4, nrow=10,
                                min=-1, max=1,
                                win=win_w1, legend='stage 1: weights', padding=1}
         win_w2 = image.display{image=model.net:get(4).weight, zoom=4, nrow=30,
                                min=-1, max=1,
                                win=win_w2, legend='stage 2: weights', padding=1}
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model.net:get(2).weight):resize(2048,1024)
         win_w1 = image.display{image=W1, zoom=0.5,
                                min=-1, max=1,
                                win=win_w1, legend='W1 weights'}
         local W2 = torch.Tensor(model.net:get(2).weight):resize(10,2048)
         win_w2 = image.display{image=W2, zoom=0.5,
                                min=-1, max=1,
                                win=win_w2, legend='W2 weights'}
      end
   end
   iter = iter + 1
end




function saveModel(model)
    local opt = model.opt
    -- save/log current net
    local filename = paths.concat(opt.save, 'model.'..os.date('%Y.%m.%d-%H:%M:%S')..'.th7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if sys.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<convnet> saving network to '..filename)
    torch.save(filename, model)
end


