function res = Centroid(TrainingMatrix,TestMatrix)
% trainingfilename = '/Users/abrar/Downloads/DataMiningProject1/ATNT50/trainDataXY.txt';
% testfilename = '/Users/abrar/Downloads/DataMiningProject1/ATNT50/testDataX.txt';
% delimiterIn = ',';
% TrainingMatrix = importdata(trainingfilename,delimiterIn);
[Header,I]=sort(TrainingMatrix(1,:));
SortedTrainingMatrix=TrainingMatrix(:,I);
SortedTrainingMatrix(1,:)=[];
Classes = unique(Header);
noOfItems = size((Header),2);
noOfUniqueClass = size(Classes,2);
noOfFeatureVector = size(SortedTrainingMatrix,1);
centroid_results = zeros(noOfFeatureVector,noOfUniqueClass);
[frequencies,b] = histc(Header,Classes);
start_i =1;
end_i = 1;
for i=1:noOfUniqueClass
    cur_class = Classes(1,i);
    end_i = start_i+(frequencies(i)-1);
    xc = mean(SortedTrainingMatrix(:,start_i:end_i),2);
    centroid_results(:,i) = xc;
    start_i= end_i+1;
end

% TestMatrix = importdata(testfilename,delimiterIn);
noOfTestColumn = size(TestMatrix,2);
result = [1,noOfTestColumn];
for i=1:noOfTestColumn
    curCol = TestMatrix(:,i);
    Q= repmat(curCol,1,noOfUniqueClass);
    E_distance = sqrt(sum((Q-(centroid_results)).^2));
    E_distance = [Classes;E_distance];
    E_distance = E_distance';
    E_distance_sort = sortrows(E_distance,2);
    topk= E_distance_sort(1,1);
    result(1,i)=topk(1,1);
end
% result = [result;TestMatrix];
res = result;
end