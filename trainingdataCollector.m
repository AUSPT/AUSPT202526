%useless now can ignore


fs = 16e3;
segSec = 2;
segSamp = segSec*fs;

win = 512; overlap = 256; nfft = 512;
w = hamming(win,'periodic');

outDir = "flac_cleanaudio";
if ~exist(outDir,'dir'), mkdir(outDir); end

k = 1;  

while hasdata(adsClean)
    [y, info] = read(adsClean);  

    
    fs0 = info.SampleRate;

    if size(y,2) > 1, y = mean(y,2); end
    if fs0 ~= fs, y = resample(y, fs, fs0); end

    nSeg = floor(size(y,1) / segSamp);

    for s = 1:nSeg
        idx1 = (s-1)*segSamp + 1;
        idx2 = idx1 + segSamp - 1;
        x = y(idx1:idx2);

        S = stft(x, fs, 'Window', w, 'OverlapLength', overlap, 'FFTLength', nfft);

        X = log1p(abs(S));   

        
        save(sprintf('X_%06d.mat',k), 'X');
        audiowrite(fullfile(outDir, sprintf("cleanSeg_%06d.flac", k)), x, fs);
        k = k + 1;
    end
end
