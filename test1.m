trainfilename = '/Users/abrar/Downloads/fileproj2/ATNTFaceImages400.txt';
% trainfilename = '/Users/abrar/Downloads/fileproj2/HandWrittenLetters.txt';
delimiterIn = ',';
k=5;

TrainData = importdata(trainfilename,delimiterIn);
noOfItems = size((TrainData),2);
noOfClass = noOfItems/k;
noOfUniqueClass = size(unique(TrainData(1,:)),2);
noOfSample = noOfItems/noOfUniqueClass;
noOfSampleTest = ceil(noOfSample/k);
TestData=[];
initialHeader = TrainData(1,:);
finalHeaderKNN = zeros(1,noOfItems);
finalHeaderLinearRegression = zeros(1,noOfItems);
finalHeaderCentroid = zeros(1,noOfItems);
finalaccuracyKNN = 0;
finalaccuracyLinearRegression=0;
finalaccuracyCentroid=0;
finalaccuracysvmLin = 0;
finalaccuracysvmGaus = 0;
% TestData = [];
for i=1:k
    TempTrainData = TrainData;
    TestData = [];
    for j=1:noOfUniqueClass
        startCol = ((j-1)*noOfSample)+((i-1)*noOfSampleTest)+1;
        endCol = startCol+noOfSampleTest-1;
        if i==k
            endCol = j*noOfSample;
        end
        TestData = horzcat(TestData,TrainData(:,startCol:endCol));
        TempTrainData(1,startCol:endCol) = 0;
    end
    indices = find(TempTrainData(1,:)==0);
    TempTrainData(:,indices)=[];
    actualClass = TestData(1,:);
    TestData(1,:)=[];
    noOfTestCases = size(TestData,2);
    fnrKNN = knn(TempTrainData,TestData);
    fnrLinearRegression = linearRegression(TempTrainData,TestData);
    fnrCentroid = Centroid(TempTrainData,TestData);
    finalResultKNN = fnrKNN-actualClass;
    finalResultLinearRegression = fnrLinearRegression-actualClass;
    finalResultCentroid = fnrCentroid-actualClass;
    differenceKNN = nnz(finalResultKNN);
    accuracyKNN = ((noOfTestCases-differenceKNN)/noOfTestCases)*100;
    differenceLinearRegression = nnz(finalResultLinearRegression);
    accuracyLinearRegression = ((noOfTestCases- differenceLinearRegression)/noOfTestCases)*100;
    differenceCentroid = nnz(finalResultCentroid);
    accuracyCentroid = ((noOfTestCases-differenceCentroid)/noOfTestCases)*100;
    fprintf('Accuracy for fold k = %i\n', i)
    fprintf('KNN %.2f\n', accuracyKNN)
    fprintf('Linear Regression %.2f\n', accuracyLinearRegression)
    fprintf('Centroid %.2f\n', accuracyCentroid)
    finalaccuracyKNN = finalaccuracyKNN+accuracyKNN;
    finalaccuracyLinearRegression = finalaccuracyLinearRegression+accuracyLinearRegression;
    finalaccuracyCentroid=finalaccuracyCentroid+accuracyCentroid;
    accuracysvmLinVec = zeros(1,size(actualClass,2));
    accuracysvmGausVec = zeros(1,size(actualClass,2));
    for m=1:noOfUniqueClass
        SVMTempTrainData = TempTrainData;
        otherClasses = find(TempTrainData(1,:)==m);
        SVMTempTrainData(1,:) = 0;
        SVMTempTrainData(1,otherClasses) = m;
        SvmTrainData = SVMTempTrainData(2:end,:);
        SvmTrainClass = SVMTempTrainData(1,:);
        SVMStructLin = svmtrain(SvmTrainData',SvmTrainClass','kernel_function','linear');
        SVMStructGaus = svmtrain(SvmTrainData',SvmTrainClass','kernel_function','rbf','RBF_Sigma',22);
        classificationLin = svmclassify(SVMStructLin,TestData');
        classificationGaus = svmclassify(SVMStructGaus,TestData');
        classificationLin = classificationLin';
        classificationGaus = classificationGaus';
        matchingLin = actualClass==classificationLin;
        matchingGaus = actualClass==classificationGaus;
        accuracysvmLinVec = accuracysvmLinVec+matchingLin;
        accuracysvmGausVec = accuracysvmGausVec+matchingGaus;
    end
    accuracysvmLin = sum(accuracysvmLinVec)*100/noOfTestCases;
    accuracysvmGaus = sum(accuracysvmGausVec)*100/noOfTestCases;
    fprintf('SVM Linear %.2f\n', accuracysvmLin)
    fprintf('SVM Gaussean %.2f\n', accuracysvmGaus)
    finalaccuracysvmLin = finalaccuracysvmLin+accuracysvmLin;
    finalaccuracysvmGaus = finalaccuracysvmGaus+accuracysvmGaus;
end
finalaccuracyKNN = finalaccuracyKNN/k;
    finalaccuracyLinearRegression = finalaccuracyLinearRegression/k;
    finalaccuracyCentroid=finalaccuracyCentroid/k;
    finalaccuracysvmLin = finalaccuracysvmLin/k;
    finalaccuracysvmGaus = finalaccuracysvmGaus/k;
    fprintf('Final Accuracy \n')
    fprintf('KNN %.2f\n', finalaccuracyKNN)
    fprintf('Linear Regression %.2f\n', finalaccuracyLinearRegression)
    fprintf('Centroid %.2f\n', finalaccuracyCentroid)
    fprintf('SVM Linear %.2f\n', finalaccuracysvmLin)
    fprintf('SVM Gaussean %.2f\n', finalaccuracysvmGaus)