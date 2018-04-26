%--------------------------------------------------------------------------
%
%  This script shows an example how to compute the No-Reference (NR) 
%  features for EPFL-PoliMi 4CIF dataset, using compute_features.m
%
%  Written by Jari Korhonen, Shenzhen University.
%
%  To use the script, EPFL-PoliMi 4CIF dataset should be downloaded and
%  the impaired sequences decoded in folder f:\\epfl-polimi. If you use
%  some other path, change the path in the script accordingly.
%

% EPFL_PoliMI 4CIF contents
sources = {'CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'};
frames = [240 240 290 230 240 290];
resolution = [704 576];

video_dir = 'f:\\epfl-polimi\\decoded\\4cif'; % Where the video files are
csv_dir = 'f:\\epfl-polimi';                  % Where the csv file goes

% Loop through all sources
for i=1:length(sources)
    
    ref_file = sprintf('%s\\%s.yuv', video_dir, sources{i});
    test_path = sprintf('%s\\%s_plr*.yuv', video_dir, sources{i});
    test_files = dir(test_path);
   
    % Loop through all the sequences for the source
    for j=1:1 %length(test_files)
        fprintf('Processing target video file: %s\n', test_files(j).name);
        
        % Compute the features for this sequence and save them in csv file
        test_file = sprintf('%s\\%s',video_dir,test_files(j).name);
        csv_file = sprintf('%s\\%s_seq_%02d.csv',csv_dir,sources{i},j);
        compute_features(test_file, ref_file, resolution, ... 
                         [1 frames(i)], csv_file);
    end
end

fprintf('Ready!!!\n');
