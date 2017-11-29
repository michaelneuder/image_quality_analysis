clear all;

% file read
orig = dlmread('/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/orig_500.txt', ' ');
recon = dlmread('/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/recon_500.txt', ' ');

% slice extra column
orig = orig(1:500,1:9216);
recon = recon(1:500,1:9216);

% initialize
image_dim = 96;
K = [0.01 0.03];
window = ones(11)/121.;
level = 3;
weight = [0.07155, 0.4530, 0.47545];
L = 255;

% reshape array
orig = reshape(orig,[500, image_dim, image_dim]);
recon = reshape(recon,[500, image_dim, image_dim]);

orig_temp = reshape(orig(1,:,:),[image_dim, image_dim]);
recon_temp = reshape(recon(1,:,:),[image_dim, image_dim]);
overall_msssim = msssim(orig_temp, recon_temp, K, window, level, weight);