clear all;

% file read
orig = dlmread('/Users/michaelhneuder/Dropbox/github/image_quality_analysis/data/sample_data/orig_140.txt', ' ');
recon = dlmread('/Users/michaelhneuder/Dropbox/github/image_quality_analysis/data/sample_data/recon_140.txt', ' ');

% slice extra column
orig = orig(1:140,1:9216);
recon = recon(1:140,1:9216);

% initialize
image_dim = 96;
% ssim = zeros(500,86,86);
ms_ssim = zeros(140,1);
K = [0.01 0.03];
window = ones(11)/121.;
level = 3;
weight = [0.20 0.30 0.50];
L = 1255;

% reshape array
orig = reshape(orig,[500, image_dim, image_dim]);
recon = reshape(recon,[500, image_dim, image_dim]);

% loop and call
for i=1:500
    orig_temp = reshape(orig(i,:,:),[image_dim, image_dim]);
    recon_temp = reshape(recon(i,:,:),[image_dim, image_dim]);
%     [mssim, ssim_map] = ssim_index(orig_temp, recon_temp, K, window);
    msssim_temp = msssim(orig_temp, recon_temp, K, window, level, weight);
    ms_ssim(i) = msssim_temp;
end
