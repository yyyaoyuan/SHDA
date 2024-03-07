function [accuracy] = Evaluate(predictedLabel,labelTestLR)
count = sum(predictedLabel == labelTestLR);
accuracy = count / size(labelTestLR,1);	
end

