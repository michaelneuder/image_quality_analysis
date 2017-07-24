%[e_u,e_s,t_u,t_s,score] = textread('rob_small_test_scores_lbpnn_whittled2.txt', '%s%s%s%s%f', 'delimiter',',');
orig = importdata('orig.txt', ' ');
recon = importdata('recon.txt',' ');
out = fopen('ssim.txt', 'w');
n_pad = 5;
for i = 1:size(orig,1)
  fprintf('%d\n',i);
  o = reshape(orig(i,:), [96 96] );
  r = reshape(recon(i,:), [96 96] );
  o_big = padarray(o, [n_pad n_pad], 'replicate');
  r_big = padarray(r, [n_pad n_pad], 'replicate');
  [mssim,ss,~] = do_ssim(o_big, r_big);
  fprintf(out,'%f ', ss);
  fprintf(out,'\n');
end
fclose(out);