clear all;
orig = dlmread('/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/orig_500.txt', ' ');
recon = dlmread('/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/recon_500.txt', ' ');
orig = orig(1:500,1:9216);
recon = recon(1:500,1:9216);
image_dim = 96;
orig = reshape(orig,[500, image_dim, image_dim]);
recon = reshape(recon,[500, image_dim, image_dim]);
ssim = zeros(500,86,86);
mssim = zeros(500,86,86);
K = [0.01 0.03];
window = ones(11);
L = 1255;
for i=1:500
    [mssim, ssim_map] = ssim_index(orig(i,:,:), recon(i,:,:));
    ssim(i,:,:) = ssim_map;
end

% fileID = fopen('test.csv','w');
% fprintf(fileID,ssim);
% fclose(fileID);
dlmwrite('test.csv', ssim);