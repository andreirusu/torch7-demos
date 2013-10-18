require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'nnd'

---[=[
trsize = 50000
tesize = 10000
format = "binary"
doubleTensor = false


-- set tensor precision
if doubleTensor then
    torch.setdefaulttensortype('torch.DoubleTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end
print('<torch> set default Tensor type to: ' .. torch.getdefaulttensortype())

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://data.neuflow.org/data/cifar-10-torch.tar.gz'
   local tar = sys.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end

print("Loading datasets...")
-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
}
for i = 0,4 do
    subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
    trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
    trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    subset = nil
    collectgarbage()
end
trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():type(torch.getdefaulttensortype()),
   labels = subset.labels[1]:type(torch.getdefaulttensortype()),
}
testData.labels = testData.labels + 1

collectgarbage()
print("Done.")

print('Preprocessing data (color space + normalization)...')
-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--


-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(11):type(torch.getdefaulttensortype()))
for i = 1,trsize do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end
-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,tesize do
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

normalization = nil
mean_u = nil
mean_v = nil 
std_u = nil
std_v = nil
collectgarbage()

for i = 0,4 do
    batch_name = 'cifar-10-batches-t7/proc.data_batch_'..(i+1) 
    subset = {
        data = trainData.data[{ {i*10000+1, (i+1)*10000} }]:clone(),
        labels = trainData.labels[{ {i*10000+1, (i+1)*10000} }]:clone()
    }
    print("Saving: "..batch_name)
    print(subset)
    torch.save(batch_name..'.t7', subset, format)
    subset = nil
    collectgarbage()
end

batch_name = 'cifar-10-batches-t7/proc.test_batch'
print("Saving: "..batch_name)
print(testData)
torch.save(batch_name..'.t7', testData, format)

trainData = nil
testData = nil
collectgarbage()

print("Done.")
--]=]
require "image"
-- write out train batches 
for i = 0,4 do
    batch_name = 'cifar-10-batches-t7/proc.data_batch_'..(i+1) 
    print("Printing: "..batch_name)
    subset = torch.load(batch_name..'.t7', format)
    print(subset)
    p = subset.data:transpose(2,1):reshape(3, 100, 100, 32, 32):transpose(3,4):reshape(3, 32*100, 32*100):transpose(1,2):reshape(32*100, 3*32*100)
    image.save(batch_name..'.png', p)
    subset = nil
    collectgarbage()
end
-- write out test batch
batch_name = 'cifar-10-batches-t7/proc.test_batch'
print("Saving: "..batch_name)
subset = torch.load(batch_name..'.t7', format)
print(subset)
p = subset.data:transpose(2,1):reshape(3, 100, 100, 32, 32):transpose(3,4):reshape(3, 32*100, 32*100):transpose(1,2):reshape(32*100, 3*32*100)
image.save(batch_name..'.png', p)
print("Done.")

