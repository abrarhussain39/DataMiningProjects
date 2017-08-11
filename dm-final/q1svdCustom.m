fileName = 'C:\Users\Dell\Documents\MATLAB\ATNTFaceImages400.txt';
% fileName = 'C:\Users\Dell\Documents\MATLAB\HandWrittenLetters.txt';
delimiterIn = ',';
MatrixData = importdata(fileName,delimiterIn);
Header = MatrixData(1,:);
Class = unique(Header);
noOfUniqueClass = size(Class,2)
k=5;
orignalAccuracy = kfold(k,MatrixData);
features = [20, 40, 60, 80, 100, 150, 200, 250, 300, 644];
Output = [];
% set(gca, 'ColorOrder', [0 0 0; 0 1 0; 0 1 1; 1 0 0], 'NextPlot', 'replacechildren');
% Output = vertcat(Output,orignalAccuracy);
% for i=1:4
% %     figure(1);
%     Yaxis=[orignalAccuracy(i),orignalAccuracy(i),orignalAccuracy(i),orignalAccuracy(i)];
%     Xaxis=[10,20,30,50];
%     plot(Xaxis,Yaxis,'--')
%     legend('Og-KNN','Og-LinearRegression','Og-Centroid','Og-SVM-Linear')
%     hold on 
% end
% outputFtest = Output;
% outputPca = Output;
% outputLdac = Output;
% outputClassic = Output;
% outputLaplace = Output;
fnctn = 1;
x=1;
switch x
    case 1  
            for i=1:10
                op=fTest(MatrixData,features(i),fnctn);
                Output=vertcat(Output,op);
                
            end
    case 2
            for i=1:10
                op=pca_svd(MatrixData,features(i),fnctn);
                Output=vertcat(Output,op);
            end
    case 3
            
                op=ldac(MatrixData,noOfUniqueClass-1,fnctn);
                Output=vertcat(Output,op);
            
    case 4
            for i=1:10
                op=classical(MatrixData,features(i),fnctn);
                Output=vertcat(Output,op);
            end          
    case 5
            for i=1:10
                op=laplace(MatrixData,features(i),fnctn);
                Output=vertcat(Output,op);
            end  
        
    otherwise
        
end
%     Output = Output';
if (x==3)
    for i=1:4
%     figure(2);
    Yaxis=[Output(2,i),Output(2,i),Output(2,i),Output(2,i)];
    Xaxis=[10,20,30,50];
    plot(Xaxis,Yaxis,'LineWidth',2)
    legend('Og-KNN','Og-LinearRegression','Og-Centroid','Og-SVM-Linear','KNN','LinearRegression','Centroid','SVM-Linear')
    hold on 
    end   
else
%     set(gca, 'ColorOrder', [0.0 0.0 0.0; 0.25 0.25 0.25; 0.5 0.5 0.5; 1 1 1], 'NextPlot', 'replacechildren');
myLegends = ['KNN','LinearRegression','Centroid','SVM-Linear'];
for j=1:4
 yaxis=Output(1:10,j);
 xaxis=[20, 40, 60, 80, 100, 150, 200, 250, 300, 644];
 figure(j);
 plot(xaxis,yaxis,'sb-')
%  legend('Og-KNN','Og-LinearRegression','Og-Centroid','Og-SVM-Linear','KNN','LinearRegression','Centroid','SVM-Linear')
if (j==1)
    legend('KNN')
end
if (j==2)
    legend('Linear Regression')
end
if (j==3)
    legend('Centroid')
end
if (j==4)
    legend('SVM Linear')
end

hold on
end

end