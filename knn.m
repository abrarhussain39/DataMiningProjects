function res = knn(TrainData,TestData)
k=3;
% TrainData = importdata(trainfilename,delimiterIn);
Header = TrainData(1,:);
TrainData(1,:)=[];
% TestData = importdata(testfilename,delimiterIn);
noOfTestColumn = size(TestData,2);
noOfTrainColumns = size(TrainData,2);
result = [1,noOfTestColumn];
for i=1:noOfTestColumn
    curCol = TestData(:,i);
    Q= repmat(curCol,1,noOfTrainColumns);
    E_distance = sqrt(sum((Q-(TrainData)).^2));
    E_distance = [Header;E_distance];
    E_distance = E_distance';
    E_distance_sort = sortrows(E_distance,2);
    topk= E_distance_sort(1:k,1);
    result(1,i)=mode(topk);
end

    res = result;
%     res
end
