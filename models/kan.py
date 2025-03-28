import torch
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,  # 输入特征的维度
            out_features,  # 输出特征的维度
            grid_size=5,  # 网格的大小（用于B样条插值）
            spline_order=3,  # B样条的阶数
            scale_noise=0.1,  # 噪声的尺度（用于初始化）
            scale_base=1.0,  # 基本权重的尺度
            scale_spline=1.0,  # B样条权重的尺度
            enable_standalone_scale_spline=True,  # 是否启用单独的B样条缩放
            base_activation=torch.nn.SiLU,  # 基本激活函数
            grid_eps=0.02,  # 网格插值的精度
            grid_range=[-1, 1],  # 网格范围
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features  # 输入特征的数量
        self.out_features = out_features  # 输出特征的数量
        self.grid_size = grid_size  # 网格的大小
        self.spline_order = spline_order  # B样条的阶数

        # 计算网格步长（网格范围除以网格大小）
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)  # 扩展到(in_features, grid_size + spline_order)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 注册网格为缓冲区，防止其成为模型的参数

        # 基本权重
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # B样条权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # 是否启用单独的B样条缩放
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # 初始化其他参数
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()  # 激活函数实例化
        self.grid_eps = grid_eps

        self.reset_parameters()  # 调用参数初始化方法

    def reset_parameters(self):
        """初始化模型的参数"""
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)  # 基本权重使用He均匀初始化
        with torch.no_grad():
            # 为B样条权重初始化噪声
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            # 使用曲线拟合的方式初始化B样条权重
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # 初始化B样条缩放参数
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_features)。

        Returns:
            torch.Tensor: B样条基函数张量，形状为(batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features  # 确保输入的维度和in_features匹配

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)  # 增加一个维度以便进行B样条计算
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # 判断x是否在网格区间内
        for k in range(1, self.spline_order + 1):
            # 计算B样条的递推公式
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算给定点的曲线插值系数。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为(batch_size, in_features, out_features)。

        Returns:
            torch.Tensor: 插值系数张量，形状为(out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # 使用最小二乘法求解插值系数
        result = solution.permute(2, 0, 1)  # 转置结果为(out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """返回经过缩放后的B样条权重"""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_features)。

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, out_features)。
        """
        assert x.size(-1) == self.in_features  # 确保输入的特征数量与in_features匹配
        original_shape = x.shape  # 保存原始的形状，以便输出时恢复
        x = x.reshape(-1, self.in_features)  # 将输入展平以适应线性层

        # 计算基础线性输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 计算B样条输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output  # 基础线性输出与B样条输出相加

        output = output.reshape(*original_shape[:-1], self.out_features)  # 恢复输出的形状
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        更新网格（使其适应数据分布）。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_features)。
            margin (float): 网格的边缘留白，避免过拟合。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # 计算B样条输出
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # 对每个特征排序并计算新的网格
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)  # 更新网格
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))  # 更新B样条权重

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        该正则化模拟了论文中的L1正则化，目标是避免过拟合。
        """
        l1_fake = self.spline_weight.abs().mean(-1)  # 计算L1正则化损失
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation  # 计算概率分布
        regularization_loss_entropy = -torch.sum(p * p.log())  # 计算熵
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )
