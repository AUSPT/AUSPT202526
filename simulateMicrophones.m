function [mic_signals, target_signal] = simulateMicrophones(target_flac, fs_t, interf_flac, fs_i)


fs = 16e3;


c = 340;
t_length = 10;


target = target_flac;
if size(target,2) > 1, target = mean(target,2); end
if fs_t ~= fs, target = resample(target, fs, fs_t); end
target = target(1 : min(numel(target), fs*t_length));
target = target / max(abs(target) + 1e-12);


interf  = interf_flac;
if size(interf,2) > 1, interf = mean(interf,2); end
if fs_i ~= fs, interf = resample(interf, fs, fs_i); end
interf = interf(1:length(target));
interf = interf / max(abs(interf) + 1e-12);

target_pos = [2.45 3.45 1.5];
interf_pos = [3.20 3.00 1.5];
mic_pos = [
    2.41 2.45 1.5;
    2.49 2.45 1.5
];


ir_target = acousticRoomResponse([4.9 4.9 4.9], target_pos, mic_pos, ...
    SampleRate=fs, SoundSpeed=c, ImageSourceOrder=0);

ir_interf = acousticRoomResponse([4.9 4.9 4.9], interf_pos, mic_pos, ...
    SampleRate=fs, SoundSpeed=c, ImageSourceOrder=0);

L = length(target);
[num_mics, ir_len] = size(ir_target);
mic_signals = zeros(L+ir_len-1, num_mics);

for m = 1:num_mics
    mic_signals(:,m) = conv(target, ir_target(m,:), 'full') + ...
                       conv(interf, ir_interf(m,:), 'full');
end

mic_signals = mic_signals(1:L, :);

target_signal = target;

end
