clear all;

% file read
orig = dlmread('/Users/michaelhneuder/Dropbox/github/image_quality_analysis/data/sample_data/orig_140.txt', ' ');
recon = dlmread('/Users/michaelhneuder/Dropbox/github/image_quality_analysis/data/sample_data/recon_140.txt', ' ');

% slice extra column
orig = orig(1:140,1:9216);
recon = recon(1:140,1:9216);
overall_msssim = zeros(140,1);

% initialize
image_dim = 96;
K = [0.01 0.03];
window = ones(11)/121.;
level = 3;
weight = [0.07155, 0.4530, 0.47545];
L = 255;

% reshape array
orig = reshape(orig,[140, image_dim, image_dim]);
recon = reshape(recon,[140, image_dim, image_dim]);

for i=1:140
    orig_temp = reshape(orig(i,:,:),[image_dim, image_dim]);
    recon_temp = reshape(recon(i,:,:),[image_dim, image_dim]);
    temp_msssim = msssim(orig_temp, recon_temp, K, window, level, weight);
    overall_msssim(i) = temp_msssim;
end