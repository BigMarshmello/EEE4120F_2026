% GROUP NUMBER:
%
% MEMBERS:
%   - Alex Hillman, HLLALE010
%   - Joab Kloppers, KLPJOA002

function run_analysis()
    % TODO1:
    % Load all the sample images from the 'sample_images' folder
    image = imread('sample_images/image_1024x1024.png');
    disp(size(image));
    image = rgb2gray(image);
    
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
    [result, elapsed_time] = inbuilt_conv2(image, Gx, Gy, 'same');

    tic
    Gx_result = my_conv2(image, Gx, 'same');
    Gy_result = my_conv2(image, Gy, 'same');
    result_manual = sqrt (Gy_result.^2+ Gx_result.^2);
    elapsed_time_manual =toc;

    fprintf('Elapsed Time: %.6f\n', elapsed_time);
    fprintf('Elapsed Time: %.6f', elapsed_time_manual);
    figure;
    subplot(1,3,1); imshow(image); title('Original');
    subplot(1,3,2); imshow(result, []); title('Edge Detection');
    subplot(1,3,3); imshow(result_manual, []); title('Edge Detection 2');
    imshow(result, []);
    
    
    
end
%% ========================================================================
%  PART 1: Manual 2D Convolution Implementation
%  ==
% TODO: Use conv2 to perform 2D convolution
% output - Convolved image result (grayscale)
function [result, elapsed_time] = inbuilt_conv2(image, Gx, Gy, padding)%Add necessary input arguments

image = double(image);
tic
Gx_result = conv2(image, Gx, padding);
Gy_result = conv2(image, Gy, padding);
result = sqrt(Gx_result.^2 + Gy_result.^2);

elapsed_time = toc;

end


% TODO: Implement manual 2D convolution using Sobel Operator(Gx and Gy)
% output - Convolved image result (grayscale)
function output = my_conv2(image, kernel, padding) %Add necessary input arguments

image = double(image);
kernel_flipped = rot90(kernel, 2);

[rows, cols] = size(image);
[k_rows, k_cols] = size(kernel);
pad = floor(k_rows/2);

image_padded = padarray(image, [pad pad], 0);
output = zeros(rows, cols);

for i =1:rows
    for j =1:cols
        patch = image_padded(i:i+k_rows-1, j:j+k_cols-1);
            output(i,j) = sum(sum(patch .* kernel_flipped));
        end
    end


end
