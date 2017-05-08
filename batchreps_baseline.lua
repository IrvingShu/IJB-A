require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
require 'paths'
require 'csvigo'
require 'image'
torch.setnumthreads(2)
local model_path = '/data/yann/Webface/baseline_320_10549_epoch_30_Bs_32_Width_4_Depth_16_Dropout_0.3.t7'
local root_path = '/data/yann/IJB-A/output1'
local folder_path = 'split1/'                      --here
local f_path = '/data/yann/IJB-A/IJB-A_11_face_images/split1.csv'  --here
local model = torch.load(model_path)
model:remove(#model)
model:add(nn.Normalize(2))
for k,v in pairs(model:findModules('cudnn.SpatialBatchNormalization')) do
    v.running_std = v.save_std
end
model:cuda()
model:evaluate()
print(model)

function centered_crop(input, crop_size)
    local w1 = math.ceil((input:size(3) - crop_size) / 2)
    local h1 = math.ceil((input:size(2) - crop_size) / 2)
    return image.crop(input, w1, h1, w1+crop_size, h1+crop_size)
end

local repsCSV = csvigo.File(paths.concat(root_path, "resize320_crop224_myModel_reps.csv"), 'w')
local bs = 2
local mean = {127.5, 127.5, 127.5}
local size = 224
local f = csvigo.load({path=f_path, mode = 'large'})
for i=1, #f do
    print(('Represent: %d/%d'):format(i, #f))
    local img = image.load(f[i][1])
     img = img * 255
     for j=1, 3 do img[j]:add(-mean[j]) end
     img = img / 128
    img = image.scale(img, size, size, 'bicubic')
    local inputs = torch.Tensor(bs, 3, 224, 224)
    local hflip_img = image.hflip(img)
    inputs[1]:copy(img:view(1,img:size(1),img:size(2),img:size(3)))
    inputs[2]:copy(hflip_img:view(1,img:size(1),img:size(2),img:size(3)))
    local img_feature = model:forward(inputs:cuda()):float()
    img_feature = img_feature:sum(1) / bs
    repsCSV:write(img_feature:squeeze():totable())
end


