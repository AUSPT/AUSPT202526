function [enhanced_signal, weights] = apply_frost_beamformer(noisy_input, fs, mic_pos, target_azimuth)
% apply_frost_beamformer performs adaptive beamforming using the phased array toolbox
%
% inputs:
%   noisy_input    : n x m matrix of audio (n samples, m mics)
%   fs             : sampling rate (hz)
%   mic_pos        : 3 x m matrix of microphone coordinates [x;y;z]
%   target_azimuth : scalar, direction of arrival in degrees (e.g., 90)
%
% outputs:
%   enhanced_signal: n x 1 vector of processed audio
%   weights        : calculated beamformer weights

    % define the sensor array
    % use conformalarray to match coordinates from simulation
    array = phased.ConformalArray('ElementPosition', mic_pos);
    
    % setup the frost beamformer
    % diagonal loading is necessary for the reverberant task to prevent
    % signal cancellation.
    
    beamformer = phased.FrostBeamformer(...
        'SensorArray', array, ...
        'SampleRate', fs, ...
        'PropagationSpeed', 340, ...
        'Direction', [target_azimuth; 0], ... % [azimuth; elevation]
        'WeightsOutputPort', true, ...        % output weights for inspection
        'FilterLength', 10, ...               % temporal taps (STAP)
        'DiagonalLoadingFactor', 1e-3);       % high loading for reverb robustness
    
    % apply the beamformer
    % the beamformer expects column vectors (time x channels).
    % we process the entire signal at once.
    
    [enhanced_signal, weights] = beamformer(noisy_input);

end