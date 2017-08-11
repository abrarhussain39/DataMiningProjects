function res = kfold(k,TrainData)
% TrainData = importdata(trainfilename,delimiterIn);
noOfItems = size((TrainData),2);
accuracyMat = [];
noOfClass = noOfItems/k;
noOfUniqueClass = size(unique(TrainData(1,:)),2);
noOfSample = noOfItems/noOfUniqueClass;
noOfSampleTest = ceil(noOfSample/k);
TestData=[];
initialHeader = TrainData(1,:);
finalHeaderKNN = zeros(1,noOfItems);
finalHeaderLinearRegression = zeros(1,noOfItems);
finalHeaderCentroid = zeros(1,noOfItems);
profKFold = [3,3,3,3,2;2,2,1,1,1;3,3,3,3,2;3,3,3,3,3];
accuracyKNN = 0;
accuracyLinearRegression=0;
accuracyCentroid=0;
accuracySVMLin = 0;
accuracySVMGaus = 0;
startIndex = 1;
endIndex =1;
sampleLeft = noOfSample;
previousSampleSize = 0;
correctKNN = 0;
correctLinearRegression = 0;
correctCentriod = 0;
correctSVMLin = 0;
correctSVMGaus = 0;
noOfTestCasesTillNow = 0;
endColTillNow = [0,14,21,35];
% TestData = [];
for i=1:k
    TempTrainData = TrainData;
    TestData = [];
    foldCurrentlyLeft = k-i+1;
    noOfSampleToBeTaken = ceil(sampleLeft/foldCurrentlyLeft);
    sampleLeft = sampleLeft-noOfSampleToBeTaken;
    for j=1:noOfUniqueClass
        startCol = endColTillNow(j)+1;
        endCol = endColTillNow(j)+profKFold(j,i);
        endColTillNow(j) = endCol;
        TestData = horzcat(TestData,TrainData(:,startCol:endCol));
        TempTrainData(1,startCol:endCol) = 0;
    end
    previousSampleSize = previousSampleSize+noOfSampleToBeTaken;
    indices = find(TempTrainData(1,:)==0);
    TempTrainData(:,indices)=[];
    actualClass = TestData(1,:);
    TestData(1,:)=[];
    noOfTestCases = size(TestData,2);
    noOfTestCasesTillNow = noOfTestCasesTillNow+noOfTestCases;
    fnrKNN = knn(TempTrainData,TestData);
    fnrLinearRegression = linearRegression(TempTrainData,TestData);
    fnrCentroid = Centroid(TempTrainData,TestData);
    finalResultKNN = fnrKNN-actualClass;
    finalResultLinearRegression = fnrLinearRegression-actualClass;
    finalResultCentroid = fnrCentroid-actualClass;
    differenceKNN = nnz(finalResultKNN);
    correctKNN = correctKNN+(noOfTestCases-differenceKNN);
    accuracyKNN = (correctKNN/noOfTestCasesTillNow)*100;
    differenceLinearRegression = nnz(finalResultLinearRegression);
    correctLinearRegression = correctLinearRegression+(noOfTestCases-differenceLinearRegression);
    accuracyLinearRegression = (correctLinearRegression/noOfTestCasesTillNow)*100;
    differenceCentroid = nnz(finalResultCentroid);
    correctCentriod = correctCentriod+(noOfTestCases-differenceCentroid);
    accuracyCentroid = (correctCentriod/noOfTestCasesTillNow)*100;
    
    ResultGaus=zeros(1,size(TestData,2));
    ResultLin=zeros(1,size(TestData,2));
    SVMmodelGaus=svmtrain(TempTrainData(1,:)',TempTrainData(2:end,:)','-s 0 -t 2 -g 0.0021');
    SVMmodelLin=svmtrain(TempTrainData(1,:)',TempTrainData(2:end,:)','-s 0 -t 0');
    [ResultGaus, accuracyGaus, ~]=svmpredict(actualClass', TestData', SVMmodelGaus);
    ResultGaus=ResultGaus';
    [ResultLin, accuracyLin, ~]=svmpredict(actualClass', TestData', SVMmodelLin);
    ResultLin=ResultLin';
    finalResultSVMGaus = ResultGaus-actualClass;
    finalResultSVMLin = ResultLin-actualClass;
    differenceSVMGaus = nnz(finalResultSVMGaus);
    correctSVMGaus = correctSVMGaus+(noOfTestCases-differenceSVMGaus);
    accuracySVMGaus = (correctSVMGaus/noOfTestCasesTillNow)*100;
    differenceSVMLin = nnz(finalResultSVMLin);
    correctSVMLin = correctSVMLin+(noOfTestCases-differenceSVMLin);
    accuracySVMLin = (correctSVMLin/noOfTestCasesTillNow)*100;


    fprintf('Accuracy after fold k = %i\n', i)
    fprintf('KNN %.5f\n', accuracyKNN)
    fprintf('Linear Regression %.5f\n', accuracyLinearRegression)
    fprintf('Centroid %.5f\n', accuracyCentroid)
    fprintf('SVM Gaussean %.5f\n', accuracySVMGaus)
    fprintf('SVM Linear %.5f\n', accuracySVMLin)
    
%     accuracysvmLinVec = zeros(1,size(actualClass,2));
%     accuracysvmGausVec = zeros(1,size(actualClass,2));
%     for m=1:noOfUniqueClass
%         SVMTempTrainData = TempTrainData;
%         otherClasses = find(TempTrainData(1,:)==m);
%         SVMTempTrainData(1,:) = 0;
%         SVMTempTrainData(1,otherClasses) = m;
%         SvmTrainData = SVMTempTrainData(2:end,:);
%         SvmTrainClass = SVMTempTrainData(1,:);
%         SVMStructLin = svmtrain(SvmTrainData',SvmTrainClass','kernel_function','linear');
%         SVMStructGaus = svmtrain(SvmTrainData',SvmTrainClass','kernel_function','rbf','RBF_Sigma',22);
%         classificationLin = svmclassify(SVMStructLin,TestData');
%         classificationGaus = svmclassify(SVMStructGaus,TestData');
%         classificationLin = classificationLin';
%         classificationGaus = classificationGaus';
%         matchingLin = actualClass==classificationLin;
%         matchingGaus = actualClass==classificationGaus;
%         accuracysvmLinVec = accuracysvmLinVec+matchingLin;
%         accuracysvmGausVec = accuracysvmGausVec+matchingGaus;
%     end
%     correctSvmLin = correctSvmLin+sum(accuracysvmLinVec);
%     accuracysvmLin = (correctSvmLin/noOfTestCasesTillNow)*100;
%     correctSvmGaus = correctSvmGaus+sum(accuracysvmGausVec);
%     accuracysvmGaus = (correctSvmGaus/noOfTestCasesTillNow)*100;
%     fprintf('SVM Linear %.5f\n', accuracysvmLin)
%     fprintf('SVM Gaussean %.5f\n', accuracysvmGaus)
%     
end
accuracyMat= horzcat(accuracyMat,accuracyKNN);
accuracyMat= horzcat(accuracyMat,accuracyLinearRegression);
accuracyMat= horzcat(accuracyMat,accuracyCentroid);
accuracyMat= horzcat(accuracyMat,accuracySVMLin);
% accuracyMat= vertcat(accuracyMat,accuracySVMGaus);
res = accuracyMat;
end
