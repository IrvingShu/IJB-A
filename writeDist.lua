--avg softmax with video pooling
require 'csvigo'
torch.setnumthreads(2)
local reps_path = '/data/yann/IJB-A/output1/'
local root_path = '/data/yann/IJB-A/'
local labels = csvigo.load({path=reps_path .. 'meta1.csv', mode='large'})
local reps = csvigo.load({path=reps_path .. 'resize320_crop224_myModel_reps.csv', mode='large'})
local pair_list = csvigo.load({path=root_path .. 'split1/verify_comparisons_1.csv', mode='large'})
assert(#labels == #reps)


function dirname(str)
    if str:match(".-/.-") then
        local name = string.gsub(str, "(.*/)(.*)", "%1")
        return name
    else
        return ''
    end
end

function basename(str)
    if str:match(".-/.-") then
        local name = string.gsub(str, "(.*/)(.*)", "%2")
        return name
    else
        return ''
    end
end


for i=1, #pair_list do
    local tmp1 = {}
    local tmp2 = {}
    local temp_id1 = tonumber(pair_list[i][1])
    local temp_id2 = tonumber(pair_list[i][2])
    local pool_frame1 = torch.zeros(512)
    local pool_frame2 = torch.zeros(512)
    local Id1_frameIdx_j = {}
    local Id2_frameIdx_j = {}

    --tmp1 includes imgs feats and avgerage frame feats 
    for j=1, #labels do
        if (tonumber(labels[j][1]) == temp_id1 and dirname(labels[j][3]) == 'img/' ) then
            temp_id1_subject = tonumber(labels[j][2])
            temp_id1_img_feature = torch.Tensor(reps[j])
            table.insert(tmp1, {j, temp_id1_subject, temp_id1_img_feature})
        elseif (tonumber(labels[j][1]) == temp_id1 and dirname(labels[j][3]) == 'frame/' ) then
            table.insert(Id1_frameIdx_j, j)
        end
    end
    if (next(Id1_frameIdx_j ) ~= nil) then
        for _,ind in pairs(Id1_frameIdx_j) do
            pool_frame1:add(torch.Tensor(reps[ind]))
        end
        table.insert(tmp1, {Id1_frameIdx_j[1], tonumber(labels[Id1_frameIdx_j[1]][2]), pool_frame1 / #Id1_frameIdx_j})
    end

    --tmp2 includes imgs feats and avgerage frame feats 
    for j=1, #labels do
        if (tonumber(labels[j][1]) == temp_id2 and dirname(labels[j][3]) == 'img/' )then
            temp_id2_subject = tonumber(labels[j][2])
            temp_id2_img_feature = torch.Tensor(reps[j])
            table.insert(tmp2, {j, temp_id2_subject, temp_id2_img_feature})
        elseif (tonumber(labels[j][1]) == temp_id2 and dirname(labels[j][3]) == 'frame/' ) then
            table.insert(Id2_frameIdx_j, j)
        end
    end
    if (next(Id2_frameIdx_j) ~= nil) then
        for _,ind in pairs(Id2_frameIdx_j) do  
            pool_frame2:add(torch.Tensor(reps[ind]))
        end
        table.insert(tmp2, {Id2_frameIdx_j[1], tonumber(labels[Id2_frameIdx_j[1]][2]), pool_frame2 / #Id2_frameIdx_j})
    end

    local is_same
    if (tmp1[1][2] == tmp2[1][2]) then
        is_same = 1 
    else 
        is_same = 0
    end

    local avg_sim = {}
    local b = torch.IntTensor():range(0,20)
    for i = 1, b:size(1) do
        local sim
        local sum_denominator = 0 
        local sum_nominator = 0
        local beta = b[i]
        for m=1, #tmp1 do
            for n=1, #tmp2 do
                local s = torch.dot(tmp1[m][3],tmp2[n][3]) / (tmp1[m][3]:norm()*tmp2[n][3]:norm())
--                local s = torch.xcorr2(tmp1[m][3]:view(512,1),tmp2[n][3]:view(512,1)):squeeze()
                local w = math.exp(beta * s)
                sum_nominator = sum_nominator + w * s
                sum_denominator = sum_denominator + w
            end
        end
        sim = sum_nominator / sum_denominator
        table.insert(avg_sim, sim)
    end

    avg_sim = torch.Tensor(avg_sim):mean()

    print(temp_id1 .. ',' .. temp_id2 .. ',' .. avg_sim .. ',' .. is_same)
end


