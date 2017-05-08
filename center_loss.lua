local CenterLoss, parent = torch.class('nn.CenterLoss', 'nn.Criterion')
function CenterLoss:__init() 
	parent__init(self)
	self.Li = torch.Tensor()
	self.gradInput = {torch.Tensor(input[1]:size()), torch.Tensor(input[2]:size())} 
end

function CenterLoss:updateOutput(input, label)	
	local x = input[1]
	local c = input[2]
	local bs = x:size(1)
	local diff = x - self.c:index(1, label:long())
	self.Li = diff:norm(2,2):pow(2)
	return self.output = self.Li:sum() / 2
end


function CenterLoss:updateGradInput(input, label)
	local bs = input[1]:size(1)
	local num_class = input[2]:size(1)

	for i=1, bs do
		self.gradInput[1][i]:copy(input[1][i] - input[2][label[i]]):type(input[1]:type()) 
	end

	for j=1, sum_class do
		local sum = 0
		local nominator = 0
		for i=1, bs do
			if (label[i] == j) then 
				sum = sum + 1
				nominator  = nominator + (input[2][j] - input[1][i])	
			end 			
		end	
		local denominator = sum + 1
		self.gradInput[2][j]:copy(nominator / denominator):type(input[2]:type())
	end

	return self.gradInput
end











