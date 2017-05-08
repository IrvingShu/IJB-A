require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
require 'paths'
require 'csvigo'
require 'image'
torch.setnumthreads(2)
local model_path = 'st10549_Acc89.5_epoch10.t7'
local root_path = '/data/yann/IJB-A/'
local folder_path = 'split1/'
local meta_path = 'verify_metadata_1.csv'
local output_path = '/data/yann/IJB-A/output1/'
local model = torch.load(model_path)
local st = model:get(1)
local baseline = model:get(2)
baseline:remove(#baseline)
baseline:add(nn.Normalize(2))
model = st:add(baseline)
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

local labelsCSV = csvigo.File(paths.concat(output_path, "myModel_labels_1.1.csv"), 'w')
local repsCSV = csvigo.File(paths.concat(output_path, "myModel_reps_1.1.csv"), 'w')
local bs = 2
local mean = {127.5, 127.5, 127.5}
local size = 224
local metaData = csvigo.load({path=root_path .. folder_path .. meta_path, mode = 'large'})
for i=2, #metaData do
    print(('Represent: %d/%d'):format(i-1, #metaData-1))
    local f = metaData[i]
    local template_id = tonumber(f[1])
    local subject_id = tonumber(f[2])
    local img_path = f[3]
    local xtl = tonumber(f[7]) 
    local ytl = tonumber(f[8])  
    local weight = tonumber(f[9]) * 1.1
    local height = tonumber(f[10]) * 1.1
    local img = image.load(root_path .. img_path, 3)
     img = img * 255
     for j=1, 3 do img[j]:add(-mean[j]) end
     img = img / 128
    if (xtl > 0 and ytl  > 0 and xtl+weight > img:size(3) and ytl+height < img:size(2)) then
        img = image.crop(img, xtl,ytl,img:size(3), ytl+height)
    elseif (xtl < 0 and ytl > 0 and xtl+weight > img:size(3) and ytl+weight<img:size(2)) then
        img = image.crop(img, 0, ytl, img:size(3), ytl+weight) 
    elseif (xtl > 0 and ytl  < 0 and xtl+weight > img:size(3) and ytl+height < img:size(2)) then
        img = image.crop(img, xtl, 0, img:size(3), ytl+height)
    elseif (xtl < 0 and ytl  < 0 and xtl+weight > img:size(3) and ytl+height < img:size(2)) then
        img = image.crop(img, 0, 0, img:size(3), ytl+height)
    elseif (xtl > 0 and ytl > 0 and ytl+height > img:size(2) and xtl+weight < img:size(3))  then
        img = image.crop(img, xtl, ytl, xtl+weight, img:size(2))
    elseif (xtl < 0 and ytl > 0 and ytl+height > img:size(2) and xtl+weight < img:size(3))  then
        img = image.crop(img, 0, ytl, xtl+weight, img:size(2))
    elseif (xtl > 0 and ytl < 0 and ytl+height > img:size(2) and xtl+weight < img:size(3))  then
        img = image.crop(img, xtl, 0, xtl+weight, img:size(2))
    elseif (xtl < 0 and ytl < 0 and ytl+height > img:size(2) and xtl+weight < img:size(3))  then
        img = image.crop(img, 0, 0, xtl+weight, img:size(2))
    elseif (xtl > 0 and ytl > 0 and xtl+weight > img:size(3) and ytl+height>img:size(2)) then
        img = image.crop(img, xtl, ytl, img:size(3), img:size(2))
    elseif (xtl < 0 and ytl > 0 and xtl+weight > img:size(3) and ytl+height>img:size(2)) then
        img = image.crop(img, 0, ytl, img:size(3), img:size(2))
    elseif (xtl > 0 and ytl < 0 and xtl+weight > img:size(3) and ytl+height>img:size(2)) then
        img = image.crop(img, xtl, 0, img:size(3), img:size(2))
    elseif (xtl < 0 and ytl < 0 and xtl+weight > img:size(3) and ytl+height>img:size(2)) then
        img = image.crop(img, 0, 0, img:size(3), img:size(2))
    elseif(xtl>0 and ytl > 0 and xtl+weight<img:size(3) and ytl+height<img:size(2)) then
        img = image.crop(img, xtl,ytl, xtl+weight, ytl+height)
    elseif(xtl<0 and ytl > 0 and xtl+weight<img:size(3) and ytl+height<img:size(2)) then
        img = image.crop(img, 0,ytl, xtl+weight, ytl+height)
    elseif(xtl>0 and ytl < 0 and xtl+weight<img:size(3) and ytl+height<img:size(2)) then
        img = image.crop(img, xtl,0, xtl+weight, ytl+height)
    elseif(xtl<0 and ytl < 0 and xtl+weight<img:size(3) and ytl+height<img:size(2)) then
        img = image.crop(img, 0,0, xtl+weight, ytl+height)
    end
    img = image.scale(img, size, size, 'bicubic')
    --img = centered_crop(img, 224)
    local inputs = torch.Tensor(bs, 3, 224, 224)
    local hflip_img = image.hflip(img)
    inputs[1]:copy(img:view(1,img:size(1),img:size(2),img:size(3)))
    inputs[2]:copy(hflip_img:view(1,img:size(1),img:size(2),img:size(3)))
    local img_feature = model:forward(inputs:cuda()):float()
    img_feature = img_feature:sum(1) / bs
    labelsCSV:write({template_id, subject_id, img_path})
    repsCSV:write(img_feature:squeeze():totable())
end


