%--------------------------------------------------------------------------
%
%  Use this function to compute the Full-Reference (FR) quality indices
%  and No-Reference (NR) features for each frame of a YUV video sequence 
%  test_video, and record the results in feature_file in CSV 
%  (comma separated) format for further processing.
%
%  Written by Jari Korhonen, Shenzhen University. For more details, see: 
%
%           J. Korhonen, "Learning-based Prediction of Packet Loss
%           Artifact Visibility in Networked Video," Proc. of QoMEX'18,
%           Sardinia, Italy, May 29-31, 2018.
%
%  Input: 
%           test_video:    Path to the test video file (YUV420 format)
%           ref_video:     Path to the reference video file (YUV420 format)
%                          for computing the FR (if you want to compute 
%                          only the NR features, you can replace this with 
%                          [] or non-existing file)
%           width, height: Resolution of the YUV video
%           frame_range:   Frames to be included (e.g. 0:100)
%           feature_file:  Path to the output file (CSV format)
%
%  Output:
%           res:           0 on success, -1 on failure
%           
%           The script appends computed quality index and features to the 
%           feature_file given as an input parameter. If reference is used,
%           the first value on each row is the NR quality index. The other
%           values are the NR features (18 in total)
% 
%

function res = compute_features(test_video, ref_video, resolution, ...
                                frame_range, feature_file)

    % Initialization                         
    width = resolution(1);
    height = resolution(2);
   
    % Define FR quality model parameters here
    % Experimentally, for 4CIF: alfa=-11.0, beta=-0.11, SP=0.02
    %                 for CIF:  alfa=-27.0, beta=-0.05, SP=0.09
    % See the QoMEX'18 paper for more details
    %
    alfa = -11.0;  
    beta = -0.11;
    SP = 0.02;
    
    if height < 300 % Assume CIF resolution
        alfa = -27.0;  
        beta = -0.05;
        SP = 0.09;
    end
    
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        res = -1;
        return;
    end

    % Try to open ref_video; if cannot, FR quality indices not computed
    use_ref = false;
    if ~isempty(ref_video)
        ref_file = fopen(ref_video,'r');
        if ref_file ~= -1
            use_ref = true;
        end
    end
         
    % Open output file for appending
    csv_file = fopen(feature_file, 'w+');
    if csv_file == -1
        fclose('all');
        fprintf('Cannot open output file.');
        res = -1;
        return;
    end
      
    % Loop through all the frames in the frame_range. Note that the script
    % returns -1 if reading beyond end of the sequence; make sure
    % frame_range is valid! 
    %
    % Note that because of temporal prediction, frame range should be from
    % 1..n-2 (or narrower), not 0..n-1!
    %
    for i = frame_range(1):frame_range(2)
        
        % Sobel filter
        H = [ 1  2  1; 0  0  0; -1 -2 -1 ]./8;

        % Read frames
        if i > frame_range(1)
            prev_YUV_frame = this_YUV_frame;
            this_YUV_frame = next_YUV_frame;
            next_YUV_frame = YUVread(test_file,[width height],i+1);
            prev_Y_sob = this_Y_sob;
            this_Y_sob = next_Y_sob;
            next_Y_sob = sqrt(imfilter(next_YUV_frame(:,:,1), H).^2 + ...
                              imfilter(next_YUV_frame(:,:,1), H').^2 );            
        else
            prev_YUV_frame = YUVread(test_file,[width height],i-1);
            this_YUV_frame = YUVread(test_file,[width height],i);
            next_YUV_frame = YUVread(test_file,[width height],i+1);
            prev_Y_sob = sqrt(imfilter(prev_YUV_frame(:,:,1), H).^2 + ...
                              imfilter(prev_YUV_frame(:,:,1), H').^2 );
            this_Y_sob = sqrt(imfilter(this_YUV_frame(:,:,1), H).^2 + ...
                              imfilter(this_YUV_frame(:,:,1), H').^2 );
            next_Y_sob = sqrt(imfilter(next_YUV_frame(:,:,1), H).^2 + ...
                              imfilter(next_YUV_frame(:,:,1), H').^2 );
        end
        
        % Test if frames read successfully
        if isempty(prev_YUV_frame) || isempty(next_YUV_frame)
            res = -1;
            fclose('all');
            fprintf('Trying to read a frame outside the range.');
            return;
        end
                
        % Compute the FR quality index (if wanted)
        if use_ref == true
            ref_YUV_frame = YUVread(ref_file,[width height],i);
            if isempty(ref_YUV_frame)
                res = -1;
                fclose('all');
                fprintf('Frame range is beyond the end of ref YUV file.');
                return;
            end
            ref_Y_sob = sqrt(imfilter(ref_YUV_frame(:,:,1), H).^2 + ...
                             imfilter(ref_YUV_frame(:,:,1), H').^2 );
            FR_index = compute_FR(this_YUV_frame, ref_YUV_frame, ...
                                  this_Y_sob, ref_Y_sob, ...
                                  width, height, alfa, beta, SP);
        end
               
        % Compute the NR features
        if i == frame_range(1)  
            
            % Compute full set of features
            NR_spat_prev = compute_NR_spatial(prev_YUV_frame, ...
                                              prev_Y_sob, ...
                                              width, height);
            NR_spat_this = compute_NR_spatial(this_YUV_frame, ...
                                              this_Y_sob, ...
                                              width, height);
            NR_temp_this = compute_NR_temporal(this_YUV_frame, ...
                                               prev_YUV_frame, ...
                                               this_Y_sob, prev_Y_sob, ...
                                               width, height, ...
                                               alfa, beta);
            NR_temp_next = compute_NR_temporal(next_YUV_frame, ...
                                               this_YUV_frame, ...
                                               next_Y_sob, this_Y_sob, ...
                                               width, height, ...
                                               alfa, beta);
            NR_features = [NR_spat_this NR_temp_this ...
                           NR_spat_prev NR_temp_next];
            
        else
            % Some features can be "recycled" from the previous round
            NR_spat_prev = NR_spat_this;
            NR_temp_this = NR_temp_next;
            NR_spat_this = compute_NR_spatial(this_YUV_frame, ...
                                              this_Y_sob, ...
                                              width, height);
            NR_temp_next = compute_NR_temporal(next_YUV_frame, ...
                                               this_YUV_frame, ...
                                               next_Y_sob, this_Y_sob, ...
                                               width, height, ...
                                               alfa, beta);
            NR_features = [NR_spat_this NR_temp_this ...
                           NR_spat_prev NR_temp_next];
        end
        
        % Write the FR index (if computed) in the CSV file
        if use_ref == true
            fprintf(csv_file, '%1.5f,', FR_index);
        end
   
        % Write the NR features in the CSV file
        for ftr = 1:length(NR_features)-1
            fprintf(csv_file, '%1.5f,', NR_features(ftr));
        end
        fprintf(csv_file, '%1.5f\n', NR_features(length(NR_features)));
        
    end
    
    fclose(test_file);
    fclose(csv_file);
    if use_ref == true
        fclose(ref_file);
    end
    res = 0;

end

%--------------------------------------------------------------------------
% This function computes the Full-Reference (FR) error visibility index
% for tar_frame, using ref_frame as a reference
%
function res = compute_FR(tar_frame, ref_frame, sob_tarY, sob_refY, ...
                          width, height, alfa, beta, SP)

    % Initialize
    block_size = 16;
    by = floor(height/block_size); 
    bx = floor(width/block_size);
    E_bl = zeros(by, bx);
    
    % Loop through all the macroblocks
    for i=1:by
        for j=1:bx
            
            % Define macroblock boundaries
            y_beg = (i-1)*block_size+1;
            x_beg = (j-1)*block_size+1;
            y_end = i*block_size;
            x_end = j*block_size;
            
            % Compute spatial activity for ref and target blocks
            S_ref = std2(sob_refY(y_beg+2:y_end-2,x_beg+2:x_end-2,1));
            S_tar = std2(sob_tarY(y_beg+2:y_end-2,x_beg+2:x_end-2,1));
            MSE = mean2((ref_frame(y_beg:y_end,x_beg:x_end,1) - ...
                         tar_frame(y_beg:y_end,x_beg:x_end,1)).^2);
      
            % Compute visibility index for each block
            S = min(S_tar, S_ref);            
            PSNR = 10*log10(1/MSE);
            E_bl(i,j)=1-1./(1+exp(alfa*S+beta*PSNR+1));
        end
    end
    
    % Spatial pooling: combine block error indices into a frame index
    E_bl = sort(E_bl(:),'descend');
    res = mean(E_bl(1:floor(bx*by*SP)));
    
end

%--------------------------------------------------------------------------
% This function computes the No-Reference (NR) spatial features for
% input frame YUV_frame
%
function features = compute_NR_spatial(YUV_frame, Y_sobeled, width, height)

    % Initialize
    block_size = 16;
    by=floor(height/block_size); 
    bx=floor(width/block_size);
    D = zeros(by,bx); % Edge discontinuties per block
    S = zeros(by,bx); % Spatial activity per block
      
    % Loop through macroblocks to compute D and S
    for i=1:by
        for j=1:bx
            
            % Block boundaries
            y_beg = (i-1)*block_size+1;
            x_beg = (j-1)*block_size+1;
            y_end = i*block_size;
            x_end = j*block_size;
                       
            % Edge discontinuity computation
            d_OB_UP = 0; d_IB_UP = 0; d_CB_UP = 0; 
            d_OB_DOWN = 0; d_IB_DOWN = 0; d_CB_DOWN = 0; 
            d_OB_LEFT = 0; d_IB_LEFT = 0; d_CB_LEFT = 0; 
            d_OB_RIGHT = 0; d_IB_RIGHT = 0; d_CB_RIGHT = 0; 
            if i>1
                d_OB_UP = mean((YUV_frame(y_beg-2,x_beg:x_end,1) - ...
                                YUV_frame(y_beg-1,x_beg:x_end,1)).^2);
                d_IB_UP = mean((YUV_frame(y_beg+1,x_beg:x_end,1) - ...
                                YUV_frame(y_beg,  x_beg:x_end,1)).^2);
                d_CB_UP = mean((YUV_frame(y_beg,  x_beg:x_end,1) - ...
                                YUV_frame(y_beg-1,x_beg:x_end,1)).^2);
            end
            if i<by
                d_OB_DOWN = mean((YUV_frame(y_end+2,x_beg:x_end,1) - ...
                                  YUV_frame(y_end+1,x_beg:x_end,1)).^2);
                d_IB_DOWN = mean((YUV_frame(y_end-1,x_beg:x_end,1) - ...
                                  YUV_frame(y_end,  x_beg:x_end,1)).^2);
                d_CB_DOWN = mean((YUV_frame(y_end,  x_beg:x_end,1) - ...
                                  YUV_frame(y_end+1,x_beg:x_end,1)).^2);
            end    
            if j>1
                d_OB_LEFT = mean((YUV_frame(y_beg:y_end,x_beg-2,1) - ...
                                  YUV_frame(y_beg:y_end,x_beg-1,1)).^2);
                d_IB_LEFT = mean((YUV_frame(y_beg:y_end,x_beg+1,1) - ...
                                  YUV_frame(y_beg:y_end,x_beg,1)).^2);
                d_CB_LEFT = mean((YUV_frame(y_beg:y_end,x_beg,1) - ...
                                  YUV_frame(y_beg:y_end,x_beg-1,1)).^2);
            end
            if j<bx
                d_OB_RIGHT = mean((YUV_frame(y_beg:y_end,x_end+2,1) - ...
                                   YUV_frame(y_beg:y_end,x_end+1,1)).^2);
                d_IB_RIGHT = mean((YUV_frame(y_beg:y_end,x_end-1,1) - ...
                                   YUV_frame(y_beg:y_end,x_end,1)).^2);
                d_CB_RIGHT = mean((YUV_frame(y_beg:y_end,x_end,1) - ...
                                   YUV_frame(y_beg:y_end,x_end+1,1)).^2);
            end              
            
            if (d_OB_UP+d_IB_UP+d_CB_UP+...
                d_OB_DOWN+d_IB_DOWN+d_CB_DOWN)>0 && ...
               (d_OB_LEFT+d_IB_LEFT+d_CB_LEFT+...
                d_OB_RIGHT+d_IB_RIGHT+d_CB_RIGHT)>0
            
                D_HOR = ((d_CB_LEFT+d_CB_RIGHT)^1.25) / ...
                        (d_OB_LEFT+d_OB_RIGHT+d_IB_LEFT+...
                        d_IB_RIGHT+d_CB_LEFT+d_CB_RIGHT);
                D_VER = ((d_CB_UP+d_CB_DOWN)^1.25) / ...
                        (d_OB_UP+d_OB_DOWN+d_IB_UP+...
                        d_IB_DOWN+d_CB_UP+d_CB_DOWN);
                D(i,j) = max(D_HOR,D_VER);
            else
                D(i,j) = 0;
            end
            
            % Spatial activity computation
            S(i,j) = sqrt(std2((Y_sobeled(y_beg+2:y_end-2, ...
                                          x_beg+2:x_end-2, 1))));
        end
    end

    % Compute frame level spatial features
    D_int = zeros(1,by);
    S1_int = zeros(1,by);
    S2_int = zeros(1,by);
    for y=1:by
        D_int(y) = std(D(y,:));
        S1_int(y) = mean(S(y,:));
        temp = zeros(1,bx-1);
        for x=1:bx-1
            temp(x) = S(y,x)-S(y,x+1);
        end
        S2_int(y) = std(temp);
    end    
    
    % Combine into spatial features
    features = zeros(1,6);
    features(1) = mean(mean(S));
    features(2) = std(S1_int);
    features(3) = std(S2_int);
    features(4) = mean(mean(D));
    features(5) = std2(D);
    features(6) = std(D_int); 
    
end

%--------------------------------------------------------------------------
% This function computes the No-Reference (NR) temporal features for
% input frame YUV_frame
%
function features = compute_NR_temporal(this_frame, ref_frame, ...  
                                        sob_tar_Y,  sob_ref_Y, ...
                                        width, height, alfa, beta)
   
    % Initialize                               
    block_size = 16;
    by=floor(height/block_size); 
    bx=floor(width/block_size);
    win = 16; % Search window size
    I0 = zeros(by,bx); % Motion intensity without compensation
    I = zeros(by,bx); % Motion intensity
    V = zeros(by,bx); % Motion velocity
    
    % Loop through macroblocks to compute I and V
    for i=1:by
        for j=1:bx
            
            % Block boundaries
            y_beg = (i-1)*block_size+1;
            x_beg = (j-1)*block_size+1;
            y_end = i*block_size;
            x_end = j*block_size;
               
            % Block matching: Four step search
            min_MSE=mean(mean((this_frame(y_beg:y_end, ...
                                          x_beg:x_end,1) - ...
                               ref_frame(y_beg:y_end, ...
                                         x_beg:x_end,1)).^2));
            x_disp = 0; 
            y_disp = 0;
            min_MSE_zero = min_MSE;
 
            if min_MSE>0
                step_y = [-2 -2 -2  0  0  2  2  2];
                step_x = [-2  0  2 -2  2 -2  0  2];
                step_idx = 1;        
                while(step_idx>0 && max(abs(x_disp),abs(y_disp))<=win)
                    min_MSE_orig = min_MSE;
                    for idx = 1:8
                        if y_beg+step_y(idx) > 0 && ...
                           y_beg+step_y(idx)+block_size < height+1 && ...
                           x_beg+step_x(idx) > 0 && ...
                           x_beg+step_x(idx)+block_size < width+1     

                            test_MSE = mean(mean( (...
                              ref_frame(y_beg+step_y(idx): ...
                                        y_beg+step_y(idx)+block_size-1,...
                                        x_beg+step_x(idx): ...
                                        x_beg+step_x(idx)+block_size-1,...
                                        1) - ...
                              this_frame(y_beg:y_end,x_beg:x_end,1)).^2));

                            if test_MSE < min_MSE
                                min_MSE = test_MSE;
                                y_disp = step_y(idx);
                                x_disp = step_x(idx);
                            end
                        end
                    end
                    if min_MSE<min_MSE_orig && step_idx==1
                        step_y = [-2 -2 -2  0  0  2  2  2]+y_disp;
                        step_x = [-2  0  2 -2  2 -2  0  2]+x_disp;
                    elseif step_idx==1
                        step_y = [-1 -1 -1  0  0  1  1  1]+y_disp;
                        step_x = [-1  0  1 -1  1 -1  0  1]+x_disp;
                        step_idx = 2;
                    else
                        step_idx = 0;
                    end
                end
            end
                        
            % Compute block level features   
            S_tar =  std2((sob_tar_Y(y_beg+2:y_end-2,x_beg+2:x_end-2)));
            S_ref0 = std2((sob_ref_Y(y_beg+2:y_end-2,x_beg+2:x_end-2)));
            S_ref =  std2((sob_ref_Y(y_beg+y_disp+2:y_end+y_disp-2, ...
                                     x_beg+x_disp+2:x_end+x_disp-2)));           
            S0 = min(S_tar, S_ref0);            
            PSNR = 10*log10(1/min_MSE_zero);
            I0(i,j) = 1-1./(1+exp(alfa*S0+beta*PSNR+1));            
            S = min(S_tar, S_ref);            
            PSNR = 10*log10(1/min_MSE);
            I(i,j) = 1-1./(1+exp(alfa*S+beta*PSNR+1));
            V(i,j) = (x_disp^2+y_disp^2)/(2*win^2);  
        end
    end
    
    % Compute frame level spatial features
    I_int = zeros(1,by);
    V1_int = zeros(1,by);
    V2_int = zeros(1,by);
    for y=1:by
        I_int(y) = std(I(y,:));
        V1_int(y) = mean(V(y,:));
        temp = zeros(1,bx-1);
        for x=1:bx-1
            temp(x) = V(y,x)-V(y,x+1);
        end
        V2_int(y) = std(temp);
    end  
    
    % Combine into temporal features
    features = zeros(1,6);
    features(1) = std(V1_int);
    features(2) = std(V2_int);
    features(3) = mean(mean(I));
    features(4) = std2(I);
    features(5) = std(I_int);
    features(6) = mean(mean(I0));
    
end

%--------------------------------------------------------------------------
% Read one frame from YUV file
%
function YUV = YUVread(f,dim,frnum)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    
    fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    
    % Read Y-component
    Y=fread(f,dim(1)*dim(2),'uchar');
    if length(Y)<dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y=cast(reshape(Y,dim(1),dim(2)),'double')./255;
    
    % Read U-component
    U=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(U)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double')./255;
    U=imresize(U,2.0);
    
    % Read V-component
    V=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(V)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double')./255;
    V=imresize(V,2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
end
    
