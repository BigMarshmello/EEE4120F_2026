% =========================================================================
% Practical 1: 2D Convolution Analysis
% =========================================================================
%
% GROUP NUMBER:7
%
% MEMBERS:
%   - Alex Hillman, HLLALE010
%   - Joab, Kloppers, KLPJOA002

%% ========================================================================
%  PART 3: Testing and Analysis
%  ========================================================================
%
% Compare the performance of manual 2D convolution (my_conv2) with MATLAB's
% built-in conv2 function (inbuilt_conv2)
function run_analysis()
    % TODO1:
    % Load all the sample images from the 'sample_images' folder
    %image = imread('sample_images/image_128x128.png');

    image128 = imread('sample_images/image_128x128.png');
    image256 = imread('sample_images/image_256x256.png');
    image512 = imread('sample_images/image_512x512.png');
    image1024 = imread('sample_images/image_1024x1024.png');
    image2048 = imread('sample_images/image_2048x2048.png');

    images = {image128,image256,image512,image1024,image2048};
    
    % TODO2:
    % Define edge detection kernels (Sobel kernel)
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [-1 -2 -1; 0 0 0; 1 2 1];
    
    % TODO3:
    % For each image, perform the following:
    %   a. Measure execution time of my_conv2
    %   b. Measure execution time of inbuilt_conv2
    %   c. Compute speedup ratio
    %   d. Verify output correctness (compare results)
    %   e. Store results (image name, time_manual, time_builtin, speedup)
    %   f. Plot and compare results
    %   g. Visualise the edge detection results(Optional)
   
    N = 5;

    file = fopen("results.txt","w");

    for k = 1:numel(images)
        times_manual = zeros(N,1);
        times_inbuilt = zeros(N,1);
        
        image = images{k};
        image = rgb2gray(image);
        [x,y]=size(image);

        %imshow(image)

        average_manual = 0.0;
        average_inbuilt = 0.0;

        result1 = my_conv2(image,Gx,Gy,'same');

        min_val = min(result1(:));
        max_val = max(result1(:));

        result1_scaled = (result1 - min_val) / (max_val - min_val);  % now in 0–1
        result1_255 = uint8(result1_scaled * 255);                  % convert to 0–255 integers
        %imshow(result1_255);

        imwrite(result1_255,"image_results/Manual"+string(x)+".png");

        for i = 1:N
            tic
            my_conv2(image,Gx,Gy,'same');
            times_manual(i) = toc;
        end

        average_manual = mean(times_manual);

        [result2,] = inbuilt_conv2(image,Gx,Gy,'same');
        
        min_val = min(result2(:));
        max_val = max(result2(:));

        result2_scaled = (result2 - min_val) / (max_val - min_val);  % now in 0–1
        result2_255 = uint8(result2_scaled * 255);                  % convert to 0–255 integers
        %imshow(result2_255);

        imwrite(result2_255,"image_results/Inbuilt"+string(x)+".png");

        for i = 1:N
            tic
            [,temp]=inbuilt_conv2(image,Gx,Gy,'same');
            times_inbuilt(i) = toc;
        end
    
        average_inbuilt = mean(times_inbuilt);
        
        speedup = average_manual/average_inbuilt;
        fprintf(file,"Size %d x %d\n",x,y);
        fprintf(file,"Average Duration for Manual: %.6f\n",average_manual);
        fprintf(file,"Average Duration for Inbuilt: %.6f\n\n", average_inbuilt);
        fprintf(file,"Speedup: %.6f\n\n", speedup);
  
    end
    fclose(file);
    
    
    
end
%% ========================================================================
%  PART 2: Built-in 2D Convolution Implementation
%  ========================================================================
%   
% REQUIREMENT: You MUST use the built-in conv2 function

% TODO: Use conv2 to perform 2D convolution
% output - Convolved image result (grayscale)
function result = inbuilt_conv2(image, Gx, Gy, padding)%Add necessary input arguments

image = double(image);
Gx_result = conv2(image, Gx, padding);
Gy_result = conv2(image, Gy, padding);
result = sqrt(Gx_result.^2 + Gy_result.^2);

end
%% ========================================================================
%  PART 1: Manual 2D Convolution Implementation
%  ========================================================================
%
% REQUIREMENT: You may NOT use built-in convolution functions (conv2, imfilter, etc.)

% TODO: Implement manual 2D convolution using Sobel Operator(Gx and Gy)
% output - Convolved image result (grayscale)

function Result = my_conv2(image, kernelx, kernely, padding) %Add necessary input arguments
    
    function output = convolve2d(image,kernel,padding) %function to perform the individual convolve
        image = double(image); %convert values to doubles
        kernel = rot90(kernel, 2); %con2 flips kernel therefore need manual to flip
        [rows, cols] = size(image);
        [k_rows, k_cols] = size(kernel);
    
        switch padding %different cases for the different padding options
            case "full"
                pad = k_rows-1; %get the required pad amount
                patch_Size = floor(k_rows/2); % get the patch offset value
                image_padded = padarray(image, [pad pad], 0,"both"); %get the padded image
    
                starting = ceil(k_rows/2); %get the starting and ending values for the loops
                ending = floor(k_rows/2);
            
                [padRows,padCols] = size(image_padded);
                output = zeros(rows+2, cols+2); %define the output matrix
                
                for i = starting:padRows-ending
                    for j = starting:padCols-ending
                        patch = image_padded(i-patch_Size:i+patch_Size,j-patch_Size:j+patch_Size);
                        output(i-1,j-1) = sum(sum(patch .* kernel)); %get the colvolution for each pixel
                    end 
                end
    
            case "same"
                pad = floor(k_rows/2);   %get required padding
                starting = ceil(k_rows/2); %get starting value for loops
                image_padded = padarray(image, [pad pad], 0,"both"); %pad the image with required 0s
            
                [padRows,padCols] = size(image_padded);
                output = zeros(rows, cols); %define the output matrix
                
                for i = starting:padRows-pad 
                    for j = starting:padCols-pad
                        patch = image_padded(i-pad:i+pad,j-pad:j+pad);
                        output(i-1,j-1) = sum(sum(patch .* kernel)); %get convolution for each pixel
                    end 
                end
    
            case "valid"
                patch_size = floor(k_rows/2);  %get offset value for patch
            
                [rows,cols] = size(image);
                output = zeros(rows-2, cols-2); %define output matrix
                
                for i = 2:rows-1 
                    for j = 2:cols-1
                        patch = image(i-patch_size:i+patch_size,j-patch_size:j+patch_size);
                        output(i-1,j-1) = sum(sum(patch .* kernel)); %get convolution for each pixel
                    end 
                end
        end
    end

    Gx = convolve2d(image,kernelx,padding); %use local function to get convolution for each kernel
    Gy = convolve2d(image,kernely,padding);
    Result = sqrt (Gy.^2+ Gx.^2); %combine the two to get the edge detection

end
