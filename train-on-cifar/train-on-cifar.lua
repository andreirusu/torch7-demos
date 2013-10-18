----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'nnd'



function setup()
    ----------------------------------------------------------------------
    -- parse command-line options
    --
    local dname,fname = sys.fpath()
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('CIFAR Training')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
    cmd:option('-network', '', 'reload pretrained network')
    cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
    cmd:option('-display', false, 'display input data and weights during training')
    cmd:option('-trainNoise', false, 'enable noise during training')
    cmd:option('-seed', 0, 'fixed input seed for repeatable experiments')
    cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
    cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
    cmd:option('-mb', 10, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
    cmd:option('-momentum', 0.95, 'momentum (SGD only)')
    cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
    cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
    cmd:option('-epochs', math.huge, 'maximum nb of training epochs')
    cmd:option('-format', "binary", 'cached data format')
    cmd:option('-double', false, 'enable double precision')
    cmd:option('-threads', 4, 'nb of threads to use')
    cmd:text()
    local opt = cmd:parse(arg)

    -- 10-class problem
    opt.classes = {'airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    
    print('<trainer> options:')
    print(opt)


    -- log results to files
    trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
    testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


    -- fix seed
    torch.manualSeed(opt.seed)
    print('<torch> set seed to: '.. opt.seed)

    -- threads
    torch.setnumthreads(opt.threads)
    print('<torch> set nb of threads to: ' .. opt.threads)

    -- set tensor precision
    if opt.double then
        torch.setdefaulttensortype('torch.DoubleTensor')
    else 
        torch.setdefaulttensortype('torch.FloatTensor')
    end
    print('<torch> set default Tensor type to: ' .. torch.getdefaulttensortype())

    return opt
end


function getModel(opt)
    local model 
    if opt.network == '' then
       -- define model to train
        model = {
            opt = opt,
            net = nn.Sequential(),
            criterion = nn.ClassNLLCriterion()
        }
        local net = model.net
       if opt.model == 'convnet' then
          ------------------------------------------------------------
          -- convolutional network
          ------------------------------------------------------------
          -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
          net:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
          net:add(nnd.Rectifier())
          net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
          -- stage 2 : filter bank -> squashing -> max pooling
          net:add(nn.SpatialConvolutionMap(nn.tables.random(16, 32, 4), 5, 5))
          net:add(nnd.Rectifier())
          net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
          -- stage 3 : standard 2-layer neural network
          net:add(nn.Reshape(32*5*5))
          net:add(nn.Linear(32*5*5, 32))
          net:add(nnd.Rectifier())
          net:add(nn.Linear(32,#opt.classes))
            net:add(nn.LogSoftMax())
          ------------------------------------------------------------

       elseif opt.model == 'mlp' then
          ------------------------------------------------------------
          -- regular 2-layer MLP
          ------------------------------------------------------------
          net:add(nn.Reshape(3*32*32))
          net:add(nn.Linear(3*32*32, 1*32*32))
          net:add(nn.Tanh())
          net:add(nn.Linear(1*32*32, #opt.classes))
            net:add(nn.LogSoftMax())
          ------------------------------------------------------------

       elseif opt.model == 'linear' then
          ------------------------------------------------------------
          -- simple linear model: logistic regression
          ------------------------------------------------------------
          net:add(nn.Reshape(3*32*32))
          net:add(nn.Linear(3*32*32,#opt.classes))
            net:add(nn.LogSoftMax())
       else
          print('Unknown model type')
          cmd:text()
          error()
       end
    else
       print('<trainer> reloading previously trained network')
       model = torch.load(opt.network)
       model.opt = opt
    end
    print('<torch> model:')
    print(model) 
    return model
end

----------------------------------------------------------------------
-- get/create dataset

batches = {5,1,2,3,4}

currentBatch = nil
currentBatchIndex = 0


function nextBatch(opt)
    local time = sys.clock() 
    currentBatch = nil
    if currentBatchIndex == #batches 
    then
        currentBatchIndex = 1
    else
        currentBatchIndex = currentBatchIndex + 1
    end
    batch_name = 'cifar-10-batches-t7/proc.data_batch_'..batches[currentBatchIndex]
    currentBatch = torch.load(batch_name..'.t7', opt.format)
    currentBatch.data = currentBatch.data:type(torch.getdefaulttensortype())
    currentBatch.labels = currentBatch.labels:type(torch.getdefaulttensortype())
    collectgarbage()
    time = sys.clock() - time
    print(string.format("<loader> new batch %s [ %.3fms ]", batch_name, time*1000))
    return currentBatch
end



-----------------------------------------------------------------------
-- define training and testing functions
--


function display(model, input)
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



function trainOnBatch(model, batch)
    if not batch then return nil end
    -- local vars
    local time = sys.clock()
    local opt = model.opt
    -- this matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(opt.classes)
    local shuffle = torch.randperm(batch.data:size(1))
    -- do one epoch
    for t = 1,batch.data:size(1),opt.mb do
        -- disp progress
        xlua.progress(t, batch.data:size(1))

        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+opt.mb-1,batch.data:size(1)) do
            -- load new sample
            local input = batch.data[shuffle[i]]
            local target = batch.labels[shuffle[i]]
            table.insert(inputs, input)
            table.insert(targets, target)
        end

        -- display?
        if opt.display then
            display(model, inputs[1])
        end
    
        -- retrieve parameters and gradients
        local parameters,gradParameters = model.net:getParameters()

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model.net:forward(inputs[i])
                          local err = model.criterion:forward(output, targets[i])
                              f = f + err

                              -- estimate df/dW
                              local df_do = model.criterion:backward(output, targets[i])
                              model.net:backward(inputs[i], df_do)

                              -- update confusion
                              confusion:add(output, targets[i])

                           end

                           -- normalize gradients and f(X)
                           gradParameters:div(#inputs)
                           f = f/#inputs

                           -- return f and df/dX
                           return f,gradParameters
                        end

          -- optimize on current mini-batch
          if opt.optimization == 'CG' then
             config = config or {maxIter = opt.maxIter}
             optim.cg(feval, parameters, config)

          elseif opt.optimization == 'LBFGS' then
             config = config or {learningRate = opt.learningRate,
                                 maxIter = opt.maxIter,
                                 nCorrection = 10}
             optim.lbfgs(feval, parameters, config)

          elseif opt.optimization == 'SGD' then
             config = config or {learningRate = opt.learningRate,
                                 weightDecay = opt.weightDecay,
                                 momentum = opt.momentum,
                                 learningRateDecay = 5e-7}
             optim.sgd(feval, parameters, config)

          elseif opt.optimization == 'ASGD' then
             config = config or {eta0 = opt.learningRate,
                                 t0 = nbTrainingPatches * opt.t0}
             _,_,average = optim.asgd(feval, parameters, config)

          else
             error('unknown optimization method')
          end
    end
    xlua.progress(batch.data:size(1), batch.data:size(1))
    confusion:updateValids()
    -- time taken
    time = sys.clock() - time
    print(string.format("<trainer> batch accuracy: %.2f%%  [ %.3fms ]",confusion.totalValid * 100, time*1000))

    -- print confusion matrix
    --print(confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    return model
end




function saveModel(model)
    local opt = model.opt
    -- save/log current net
    local filename = paths.concat(opt.save, 'cifar.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if sys.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, model)
end



function testOnBatch(model, batch)
    if not batch then return nil end
    -- local vars
    local time = sys.clock()
    local opt = model.opt
    -- this matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(opt.classes)
    -- averaged param use?
    if average then
        cachedparams = parameters:clone()
        parameters:copy(average)
    end
    if opt.display then     
        display(model, batch.data[1])
    end


    -- test over given batch
    for t = 1,batch.data:size(1) do
        -- disp progress
        xlua.progress(t, batch.data:size(1))

        -- get new sample
        local input = batch.data[t]
        local target = batch.labels[t]
                -- test sample
        local pred = model.net:forward(input)
        confusion:add(pred, target)
    end
    xlua.progress(batch.data:size(1), batch.data:size(1))
    print(confusion)
    
    -- timing
    time = sys.clock() - time
   print(string.format("<tester> batch accuracy: %.2f%%  [ %.3fms ]",confusion.totalValid * 100, time*1000))
    
    -- display?
    if opt.display then
        image.display(confusion:render())
    end
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

    -- averaged param use?
    if average then
        -- restore parameters
        parameters:copy(cachedparams)
    end
    return model
end



function trainModel(model)
    local time = sys.clock() 
    -- epoch tracker
    epoch = 0
    local opt = model.opt
    testOnBatch(model, nextBatch(opt))

    while true do
        -- next epoch
        epoch = epoch + 1
        if epoch > opt.epochs then 
            break 
        end
        print("\n<trainer> epoch # " .. epoch .. ' [mb = ' .. opt.mb .. ']')
        
        -- train model and save
        trainOnBatch(model, nextBatch(opt))
        trainOnBatch(model, nextBatch(opt))
        trainOnBatch(model, nextBatch(opt))
        trainOnBatch(model, nextBatch(opt))
        saveModel(model) 

        -- test model
        testOnBatch(model, nextBatch(opt))

        -- plot errors
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
        --trainLogger:plot()
        --testLogger:plot()
        print("<trainer> finished epoch # "..epoch.."\n")
    end

    print(string.format("<trainer> success! [ %.3fh ]", (sys.clock() - time)/3600))
end


function main()
    local opt = setup()
    local model = getModel(opt)
    trainModel(model)
end


do 
    main()
end

