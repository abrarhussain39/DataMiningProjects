trainfilename = '/Users/abrar/Downloads/fileproj2/ATNTFaceImages400.txt';
% trainfilename = '/Users/abrar/Downloads/fileproj2/HandWrittenLetters.txt';
TrainData = importdata(trainfilename,delimiterIn);
noOfUniqueClass = size(unique(TrainData(1,:)),2);
idx = kmeans(TrainData(2:end,:)',noOfUniqueClass,'Replicates',10);
[C,order] = confusionmat(TrainData(1,:),idx');
[assignment,cost] = munkres(~C);
orderedMat = C(:,assignment);
% orderedMat
correctResult = sum(diag(orderedMat));
accuracy = correctResult/size(TrainData,2)*100;
accuracy