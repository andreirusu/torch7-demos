require "torch"
require "torch-env"
require "sys"
require "nn"
require "nndx"

----------------------------------------------------------------------
----------       YUV, normalized & cached CIFAR10 loader     ---------
----------------------  see caching script!  -------------------------
----------------------------------------------------------------------

-- 10-class problem

function getClassLabels() 
    return {'airplane', 
            'automobile', 
            'bird', 
            'cat',
            'deer', 
            'dog', 
            'frog', 
            'horse', 
            'ship', 
            'truck'}
end


--- the batches will be used in this order 
local batches = {5, 1, 2, 3, 4}
--- use the  
local validation  = {[5] = true }


local currentBatch = nil
local currentBatchIndex = 0


local function loadBatchName(batch_name, opt)
    local time = sys.clock() 
    currentBatch = torch.load(batch_name..'.t7', opt.format)
    currentBatch.data = currentBatch.data:type(torch.getdefaulttensortype())
    if opt.distort and not opt.test then 
        local distort = nndx.Distort()
        local data = currentBatch.data
        for i=1,data:size(1) do
            data[i]:copy(distort:forward(data[i]))
        end
    end

    currentBatch.labels = currentBatch.labels:type(torch.getdefaulttensortype())
    collectgarbage()
    time = sys.clock() - time
    print(string.format("<loader> new batch %s [ %.3fms ]", batch_name, time*1000))
    return currentBatch
end

function nextBatch(opt)
    if opt.test then
        batch_name = 'cifar-10-batches-t7/proc.test_batch'
        return loadBatchName(batch_name, opt)
    end
    currentBatch = nil
    if currentBatchIndex == #batches then
        currentBatchIndex = 1
    else
        currentBatchIndex = currentBatchIndex + 1
    end

    batch_name = 'cifar-10-batches-t7/proc.data_batch_'..batches[currentBatchIndex]
    return loadBatchName(batch_name, opt),  validation[batches[currentBatchIndex]]
end

