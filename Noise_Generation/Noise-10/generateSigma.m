function A = generateSigma(n)
    % 生成随机矩阵
    B = randn(n);

    % 使其对称化
    B = (B + B') / 2;

    % 计算 B 的特征值和特征向量
    [V, D] = eig(B);

    % 将负特征值设为零以确保半正定性
    D(D < 0) = 0;

    % 重构半正定矩阵
    A = V * D * V';
end