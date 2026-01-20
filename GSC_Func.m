%function-ised (not sure that is a word??) version of GSC to generate test data spectrogram 
% 

function [spectro_gsc, Y_gsc_stft, f, t] = GSC_Func(mic_signals) 

fs=16e3;
if size(mic_signals,2) ~= 2, error('mic_signals must be N×2'); end
%commented out for DL training, unneccesary for now
% array geometry & alignment
%d = 0.08;           % mic spacing (m)
%c = 340;            % speed of sound (m/s)
%mic_locs = [-d/2, d/2];
%theta_target = 90;  % target direction

% stft setup
win_len = 512;
overlap = 256;
fft_len = 512;

[s, f, t] = stft(mic_signals, fs, 'window', hamming(win_len), 'overlaplength', overlap, 'fftlength', fft_len);
[num_bins, num_frames, ~] = size(s);

% output buffer
Y_gsc_stft = complex(zeros(num_bins, num_frames));


% frequency domain GSC processing
for k = 1:num_bins
    %freq = f(k);
    
    % pre-steering (alignment)
    % we phase-shift the inputs so the target appears to come from 0 deg (broadside).
    % this simplifies the blocking matrix significantly.
    
    %k_wave = 2*pi*freq/c;
    steering_vec=1; %currently line gbelow is redundant
    %steering_vec = exp(1j * k_wave * mic_locs' * cosd(theta_target));
    
   
        
    % apply phase shift to align signals
    X_k = squeeze(s(k, :, :)).'; % (Mics x Time)
    X_aligned = X_k ./ steering_vec; % Element-wise division removes delay
    
    % the upper rail (fixed beamformer)
    % since signals are aligned, a simple average preserves the target.
    d_upper = mean(X_aligned, 1); % (1 x Time) - "desired signal + noise"
    
    % the lower rail (blocking matrix)
    % we need to block the target. since target signals are identical in x_aligned,
    % subtracting them (mic 1 - mic 2) cancels the target perfectly.
    % in gsc terms, blocking matrix b = [1; -1].
    
    % this contains only interference + noise (Target is gone)
    u_lower = X_aligned(1,:) - X_aligned(2,:); 
    
    % adaptive cancellation (unconstrained)
    % we want to find a weight 'w' such that: output = d_upper - w * u_lower
    % is minimized.
    % w = e[u * d'] / e[u * u']  (cross-variance / auto-variance)
    
    % calculate statistics (averaging over time frames)
    R_du = (u_lower * d_upper') / num_frames; % cross-correlation
    R_uu = (u_lower * u_lower') / num_frames; % auto-correlation
    
    % robustness: add tiny noise to denominator to avoid divide-by-zero
    R_uu = R_uu + 1e-6 * mean(abs(u_lower).^2); 
    
    % calculate scalar weight
    w_adapt = R_du / R_uu; 
    
    % step e: final subtraction
    Y_gsc_stft(k, :) = d_upper - (w_adapt' * u_lower);
end

spectro_gsc = log1p(abs(Y_gsc_stft));   % size: [F x T]



% % inverse STFT
% result_GSC = istft(Y_gsc_stft, fs, 'Window', hamming(win_len), 'OverlapLength', overlap, 'FFTLength', fft_len);
% 
% % length matching
% len = min(length(mic_signals), length(result_GSC));
% result_GSC = result_GSC(1:len);

% normalize
%result_GSC = y_gsc / max(abs(y_gsc)) * 0.9; allow variability for U-net

end
