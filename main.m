%mostly variables and file directory setup can ignore.
%scroll for u net code


baseOut = "C:\Users\tjmik\OneDrive\Desktop\2025\Signal Processing Cup\AUSPT202526";

targetDir = "C:\Users\tjmik\OneDrive\Desktop\2025\Signal Processing Cup\AUSPT202526\flac_cleanaudio";                 % where you wrote cleanSeg_*.flac
targetAds = audioDatastore(targetDir, "FileExtensions",".flac");
N = numel(targetAds.Files);


mic_pos = [
    2.41 2.45 1.5;
    2.49 2.45 1.5
];

src_target_pos = [2.45; 3.45; 1.5];

src_interf_pos = [3.22; 3.06; 1.5];

fs = 16e3;
Nt = 2*fs;



[interf_full, fs_i] = audioread('interference_signal1.flac');
if size(interf_full,2)>1, interf_full = mean(interf_full,2); end
if fs_i ~= fs, interf_full = resample(interf_full, fs, fs_i); end

Ni = size(interf_full,1);
if Ni < Nt
    reps = ceil(Nt/Ni);
    interf_full = repmat(interf_full, reps, 1);
    Ni = size(interf_full,1);
end


% stftTargetDir = "C:\Users\tjmik\OneDrive\Desktop\2025\Signal Processing Cup\AUSPT202526\stft_cleanaudio";
% 
% outXgscDir = fullfile(baseOut, "Xgsc");
% outMaskDir = fullfile(baseOut, "YMask");
% 
% 
% if isempty(dir(fullfile(outXgscDir,"Xgsc_*.mat"))) || isempty(dir(fullfile(outMaskDir,"Ymask_*.mat")))
% 
% 
% 
%     for k = 1:N 
%         Xclean = load(fullfile(stftTargetDir, sprintf("X_%06d.mat",k))).X;
%         [target_flac, fs_t] = audioread(targetAds.Files{k});
%         if fs_t ~= fs
%             target_flac = resample(target_flac, fs, fs_t);
%         end
% 
%         target_flac = target_flac(1:min(end,Nt),:);
% 
% 
%         if size(target_flac,1) < Nt
%             target_flac(end+1:Nt,:) = 0;
%         end
% 
% 
% 
% 
%         if size(target_flac,2) > 1, target_flac = mean(target_flac,2); end
% 
% 
%         %random start 
%         startIdx = randi(Ni - Nt + 1);
%         interf_flac = interf_full(startIdx:startIdx+Nt-1, :);
% 
%         [mic_signals, target_signal] = simulateMicrophones(target_flac, 16e3, interf_flac,16e3);
% 
%         [Xgsc, Y_gsc_stft, f, t] = GSC_Func(mic_signals);
% 
%         mag_clean = expm1(Xclean);     
%         mag_gsc   = abs(Y_gsc_stft);   
%         Ymask     = mag_clean ./ (mag_gsc + 1e-8); 
%         Ymask     = min(max(Ymask,0),1); 
% 
%         Xgsc = single(reshape(Xgsc, size(Xgsc,1), size(Xgsc,2), 1));
%         Ymask = single(reshape(Ymask, size(Ymask,1), size(Ymask,2), 1));
% 
% 
%         save(fullfile(outXgscDir,  sprintf("Xgsc_%06d.mat",k)), "Xgsc");
%         save(fullfile(outMaskDir,  sprintf("Ymask_%06d.mat",k)), "Ymask");
% 
% 
% 
% 
% 
% 
%     end
% end   

Xfiles = dir(fullfile(outXgscDir, "Xgsc_*.mat"));
Yfiles = dir(fullfile(outMaskDir, "Ymask_*.mat"));

Xpaths = sort(fullfile({Xfiles.folder}, {Xfiles.name})');
Ypaths = sort(fullfile({Yfiles.folder}, {Yfiles.name})');

assert(numel(Xpaths) == numel(Ypaths), "Mismatch: Xgsc and Ymask counts differ.");






readY = @(f) load(f,"Ymask");

dsX = fileDatastore(Xpaths, "ReadFcn", @(f) load(f,"X").X(:,1:120,:));
dsY = fileDatastore(Ypaths, "ReadFcn", @(f) readY(f).Ymask(:,1:120,:));
dsTraining = combine(dsX, dsY);


Nall = numel(Xpaths);
idx = randperm(Nall);

Ntrain = floor(0.9 * Nall);
trainIdx = idx(1:Ntrain);
valIdx   = idx(Ntrain+1:end);

dsTrain = subset(dsTraining, trainIdx);
dsVal   = subset(dsTraining, valIdx);







%actual U-net architecture

spectrogramSize = [512 120 1];
 
encoderDepth    = 3;

       
net = unet(spectrogramSize, 2, EncoderDepth=encoderDepth);

net = removeLayers(net, "FinalNetworkSoftmax-Layer");   

regressionConv = convolution2dLayer(1, 1, ...
    Name="FinalRegressionConv", ...
    Padding="same");

net = replaceLayer(net, "encoderDecoderFinalConvLayer", regressionConv); %switch it from image classification to a regression

net.Layers(end-5:end)
net.OutputNames    

lgraph = layerGraph(net);  

lgraph = addLayers(lgraph, sigmoidLayer("Name","maskSigmoid"));
lgraph = addLayers(lgraph, regressionLayer("Name","maskLoss"));

lgraph = connectLayers(lgraph, "FinalRegressionConv", "maskSigmoid");
lgraph = connectLayers(lgraph, "maskSigmoid", "maskLoss");

options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-3, ...
    "MaxEpochs", 10, ...
    "MiniBatchSize", 16, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", dsVal, ...
    "ValidationFrequency", 50, ...
    "Plots", "training-progress", ...
    "Verbose", true,"ExecutionEnvironment","gpu");

trainedNet = trainNetwork(dsTrain, lgraph, options);
save(fullfile(baseOut, "trained_mask_unet.mat"), "trainedNet");





figure;
plot(lgraph);
title("U-Net for Spectrogram Regression");

net.Layers(end-5:end)
net.OutputNames