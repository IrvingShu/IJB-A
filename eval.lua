require 'csvigo'
local lapp = require 'pl.lapp'
local opt = lapp[[
--path  (default 'res.csv')
]]
f = csvigo.load({path=opt.path, mode='large'})
local sim =torch.FloatTensor(#f)
local label = torch.IntTensor(#f)
for i = 1, #f do
    sim[i] = tonumber(f[i][3])
    label[i] = tonumber(f[i][4])
end

function calculate_val_far(threshold, sim, label)
    local predict_same = sim:gt(threshold):int()
    local ta = 0
    local fa = 0
    local fn = 0
    local tn = 0
    for i = 1, predict_same:size(1) do
        if (predict_same[i] == 1 and label[i] == 1) then 
            ta = ta + 1
        elseif (predict_same[i] == 1 and label[i] == 0) then
            fa = fa + 1
        elseif (predict_same[i] == 0 and label[i] == 1) then
            fn = fn + 1
        elseif (predict_same[i] == 0 and label[i] == 0) then
            tn = tn + 1
        end
    end
    local n_same = label:sum()
    local n_diff = label:size(1) - label:sum()
    local tar = ta / n_same
    local far = fa / n_diff
    local acc = (ta+tn) / predict_same:size(1)
    return tar, far, acc
end


for threshold = 0, 1, 0.01 do
    tar, far, acc = calculate_val_far(threshold, sim, label)
    print("tar:" .. tar .. ', far: ' .. far .. ', acc: ' .. acc .. ', threshold:' .. threshold )
end


