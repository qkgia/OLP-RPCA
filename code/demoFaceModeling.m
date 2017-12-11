function demoFaceModeling (output_dir, saveImgFiles)

%% Load data
load('data/exp4_frontface_yaleB.mat');

startSubjIdx = 9;
nSubj = 9;
X = zeros(64, 4096);
W = ones(64, 4096);

%% Set parameters
opts = [];
opts.tol = 1e-3;
opts.maxit = 100;
opts.p = 0.01;
opts.mu_t = 5;

% for each subj
for i = startSubjIdx:nSubj

    %% Get all 64 faces into one data matrix
    X(:, :) = I(i, :, :);
    X = NormalizeImage(X);

    %%
    [out.LowRank, out.Sparse, out.iter] = nrpca(X', opts);

    %% Save results
    for j = 1:64

        %% get resulting images
        img_in = NormalizeImage(reshape(X(j, :)', 64, 64));
        img_LR = NormalizeImage(reshape(out.LowRank(:, j), 64, 64));
        img_SP = NormalizeImage(reshape(out.Sparse(:, j), 64, 64));

        if(saveImgFiles)
            if( j == 14 || j == 18 || j ==21 || j == 64)
                %% save input imaes
                imwrite(img_in, [output_dir '/p_' num2str(i, '%02d') '_l_' num2str(j) '_input.png']);

                %% save low-rank images
                imwrite(img_LR, [output_dir '/p_' num2str(i, '%02d') '_l_' num2str(j) '_LR.png']);

                %% save error images
                imwrite(img_SP, [output_dir '/p_' num2str(i, '%02d') '_l_' num2str(j) '_SP.png']);
            end
        end
    end
end