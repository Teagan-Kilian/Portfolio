clear 
close all
clc
% Digital Image Processing and Analysis Final Project
% By Teagan Kilian, Tobi Ajagbi, and Etebong Emmanuel Ibekwe

% initialization
sumscore = 0;
imnum = 0;

% applies the blur/sharp separation algorithm to each image in the range
% specified and returns the average score of the algorithm
for im =1:111 % 110 images total

    % counts the number of images
    % there is no imagge 87 so it should not be included in the number of images
    if im == 87 
        immum = imnum + 0;
    else 
        imnum = imnum + 1;
    end 

    % call the image correctly from within the loop
    if im < 10 
        im_num = ['00' num2str(im) '.jpg'];
        im_numGT = ['00' num2str(im) '.png'];
    elseif (10 <= im) && (im <= 86)  
        im_num = ['0' num2str(im) '.jpg'];
        im_numGT = ['0' num2str(im) '.png'];
    % skip the 87th loop
    elseif im == 87
        continue;
    elseif (88 <= im) && (im <= 99)  
        im_num = ['0' num2str(im) '.jpg'];
        im_numGT = ['0' num2str(im) '.png'];
    else 
        im_num = [num2str(im) '.jpg'];
        im_numGT = [num2str(im) '.png'];
    end

    % IMAGE MANIPULATION
    img = imread(fullfile('test_dataset', im_num)); % reqd the image
    img_gray = rgb2gray(img); % convert to grayscale

    % use a filter to blur the image
    h = fspecial('sobel'); % sobel filter was chosen because it gave the best results on average
    blurred_img = imfilter(img_gray, h);  % Apply the filter to the image

    % define a structuring element
    SE = strel('disk',20); % disk shape size 18
    
    % IMAGE CLOSURE (1)
    dia = imdilate(blurred_img,SE); % dialate the image
    ero = imerode(dia,SE); % erode the dialation

    % Mean Filter
    h = fspecial('average', [11,11]);  % creates the convolution term
    blurred_img1 = imfilter(ero, h);  % Apply filter to image

    % Median Filter
    median = medfilt2(blurred_img1, [15,15]);
    
    % IMAGE CLOSURE (2)
    dia1 = imdilate(median,SE); % dialate the erosion
    ero1 = imerode(dia1,SE); % erode the dialation

    % Mean Filter with smaller kernel
    h = fspecial('average', [5,5]);  % creates the convolution term
    blurred_img1 = imfilter(ero1, h);  % Apply filter to image

    % Median Filter
    blurred_img1 = medfilt2(blurred_img1, [30,30]);

    % IMAGE CLOSURE (3)
    dia2 = imdilate(blurred_img1,SE);
    ero2 = imerode(dia2,SE);

    % Convert the image to binary using thresholding
    % the threshold of 40 was chosen because it provided the best results
    newVal = ero2 > 45; 

    % Read the corresponding ground truth image
    truth = imread(fullfile('ground_truth', im_numGT));
    truth2 = double(truth);
    final = double(newVal);

    % ENSURE IMAGE AND GROUND TRUTH ARE SAME SIZE
    sizetruth = size(truth2); % size of ground truth image
    areatruth = sizetruth(1)*sizetruth(2); % area of ground truth image
    
    sizefinal = size(final); % size of manipulated image
    areafinal = sizefinal(1)*sizefinal(2); % area of ground truth image

    % test which image is smaller then resize the larger image to match the
    % smaller image to preserve quality 
    if areatruth < areafinal
        final = imresize(final, sizetruth);
    elseif sizefinal < sizetruth 
        truth2 = imresize(truth2, sizefinal);
    end

    % SCORING
    %intersectino of the final image and ground truth 
    inter = truth2 & final;
    intersum = sum(sum(truth2 & final)); % sum all the values in the intersection

    % union of the final image and the ground truth 
    uni = truth2 | final;
    unisum = sum(sum(truth2 | final)); % sum all the values in the union 

    % calculate the score by intersection / union for eqch image
    score = intersum/unisum; 
    % create a running total of the scores of each image for this method 
    sumscore = sumscore+score; 
end 

% average score of the algorithm by dividing by the number of images in the test set 
% displayed as a percentage
avgscore = (sumscore/imnum) * 100

