require "sys"
require "torch"
require "torch-env"
require "nn"
require "nndx"
require "lanes"

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
local cachedBatch = nil
local cachedBatchIndex = 1
local currentBatchIndex = 0


local function loadBatchName(batchName, opt)
    currentBatch = torch.load(batchName..'.t7', opt.format)
    currentBatch.data = currentBatch.data:type(torch.getdefaulttensortype())
    if opt.distort and not opt.test then 
        local distort = nndx.Distort()
        local data = currentBatch.data
        for i=1,data:size(1) do
            data[i]:copy(distort:forward(data[i]))
        end
    end

    currentBatch.labels = currentBatch.labels:type(torch.getdefaulttensortype())
    return currentBatch
end

cacheNextBatch = lanes.gen(function (batchName, opt)
                                        return loadBatchName(batchName, opt)
                                    end)
function nextBatch(opt)
    local time = sys.clock() 
    if opt.test then
        batchName = 'cifar-10-batches-t7/proc.test_batch'
        return loadBatchName(batchName, opt)
    end
    
    -- cache next batch
    currentBatchIndex =  cachedBatchIndex
    cachedBatchIndex = (currentBatchIndex + 1) % #batches  
    
    local batchName = 'cifar-10-batches-t7/proc.data_batch_'..batches[batchIndex]
    local cachedBatchName = 'cifar-10-batches-t7/proc.data_batch_'..batches[cachedBatchIndex]

    -- return cached batch if available
    local b
    if not cachedBatch then
        cachedBatch = cacheNextBatch(batchName, opt)  
    end
    b = cachedBatch[1]
    cachedBatch = cacheNextBatch(cachedBatchName, opt)  
    collectgarbage()
    time = sys.clock() - time
    print(string.format("<loader> new batch %s [ %.3fms ]", batchName, time*1000))
    return b,validation[batches[currentBatchIndex]]
end

