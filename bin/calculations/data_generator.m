clear all;

% file read
orig = dlmread('/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/orig_140.txt', ' ');
recon = dlmread('/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/recon_140.txt', ' ');

% slice extra column
orig = orig(1:140,1:9216);
recon = recon(1:140,1:9216);

% initialize
image_dim = 96;
ssim = zeros(140,86,86);
K = [0.01 0.03];
window = ones(11)/121.;
L = 1255;

% reshape array
orig = reshape(orig,[140, image_dim, image_dim]);
recon = reshape(recon,[140, image_dim, image_dim]);

% loop and call
for i=1:140
    orig_temp = reshape(orig(i,:,:),[image_dim, image_dim]);
    recon_temp = reshape(recon(i,:,:),[image_dim, image_dim]);
    [mssim, ssim_map] = ssim_index(orig_temp, recon_temp, K, window);
    ssim(i,:,:) = ssim_map;
end

disp(size(ssim));