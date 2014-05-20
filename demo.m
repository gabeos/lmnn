% run_lmnn
clear
close all
install;
setpaths;
figure;
clc;
rand('seed',1);

% Load data 
% load data/wine.mat;
%load data/segment.mat;
%load data/usps.mat;
fprintf('\nLoading feature files...')
datadir='./data/';
fprintf('\nData Directory: %s',datadir)
s = what(datadir);
featureFileMask=strncmp(s.mat,'dialogue-full',13);
featureFiles = s.mat(featureFileMask);
labelFileMask = strncmp(s.mat,'dialogue-labels',15);
labelFiles = s.mat(labelFileMask);
for a=1:numel(featureFiles)
    disp(strcat('Loading feature file: ',featureFiles(a)))
    load(char(strcat(datadir,featureFiles(a))))
end
for a=1:numel(labelFiles)
    disp(strcat('Loading label file: ',labelFiles(a)))
    load(char(strcat(datadir,labelFiles(a))))
end

fprintf('\tDone.\nConcatenating features...')
whoarr = who('features*');
features = zeros(0,size(eval(whoarr{1}),2));
for fn=0:9
    features = double([features;eval(strcat('features',num2str(fn)))]);
end
fprintf('\tDone.\n')

labelsD=fix(double(labels(1:size(features,1))));

trainSize=floor(size(features,1)*0.8);
validationSize=floor((size(features,1)-trainSize)/2);
testSize=(size(features,1) - trainSize - validationSize);

fprintf('Setting train and test sets...')
xTr=features(1:trainSize,:)';
xVa=features(trainSize+1:trainSize+validationSize,:)';
xTe=features(trainSize+validationSize+1:size(features,1),:)';
yTr=labelsD(1:trainSize);
yVa=labelsD(trainSize+1:trainSize+1+validationSize);
yTe=labelsD(trainSize+validationSize+1:size(labelsD,2));
fprintf('\tDone.\n')

% KNN classification error before metric learning  
errRaw=knnclassifytreeomp([],xTr, yTr,xTe,yTe,1);fprintf('\n');

%PCA
fprintf('\n')
L0=pca(xTr)';
errRaw3d=knnclassifytreeomp(L0,xTr, yTr,xTe,yTe,1);fprintf('\n');
subplot(3,2,1);
scat(L0*xTr,3,yTr);
title(['PCA Training (Error: ' num2str(100*errRaw3d(1),3) '%)'])
noticks;box on;
subplot(3,2,2);
scat(L0*xTe,3,yTe);
title(['PCA Test (Error: ' num2str(100*errRaw3d(2),3) '%)'])
noticks;box on;
drawnow


% Call LMNN to get the initiate linear transformation
fprintf('\n')
disp('Learning initial metric with LMNN ...')
[L,~] = lmnn2(xTr, yTr,3,L0,'maxiter',1000,'quiet',1,'mu',0.5,'validation',0.2,'earlystopping',25);
% KNN classification error after metric learning using LMNN
errL=knnclassifytreeomp(L,xTr, yTr,xTe,yTe,1);fprintf('\n');

% Plotting LMNN embedding
subplot(3,2,3);
scat(L*xTr,3,yTr);
title(['LMNN Training (Error: ' num2str(100*errL(1),3) '%)'])
noticks;box on;
drawnow
subplot(3,2,4);
scat(L*xTe,3,yTe);
title(['LMNN Test (Error: ' num2str(100*errL(2),3) '%)'])
noticks;box on;
drawnow


% Gradient boosting
fprintf('\n')
disp('Learning nonlinear metric with GB-LMNN ... ')
embed=gb_lmnn(xTr,yTr,3,L,'ntrees',200,'verbose',false,'XVAL',xVa,'YVAL',yVa);

% KNN classification error after metric learning using gbLMNN
errGB=knnclassifytreeomp([],embed(xTr), yTr,embed(xTe),yTe,1);fprintf('\n');
subplot(3,2,5);
scat(embed(xTr),3,yTr);
title(['GB-LMNN Training (Error: ' num2str(100*errGB(1),3) '%)'])
noticks;box on;
drawnow
subplot(3,2,6);
scat(embed(xTe),3,yTe);
title(['GB-LMNN Test (Error: ' num2str(100*errGB(2),3) '%)'])
noticks;box on;
drawnow

disp(['1-NN Error for raw (high dimensional) input is : ',num2str(100*errRaw(2),3),'%']);
disp(['1-NN Error after PCA in 3d is : ',num2str(100*errRaw3d(2),3),'%']);
disp(['1-NN Error after LMNN in 3d is : ',num2str(100*errL(2),3),'%']);
disp(['1-NN Error after gbLMNN in 3d is : ',num2str(100*errGB(2),3),'%']);

