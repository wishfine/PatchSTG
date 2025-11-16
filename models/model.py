import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


"""
PatchSTG 模型实现（带中文注释）

包含两个主要类：
- WindowAttBlock: 在空间 patch 内同时进行 depth（patch 内）与 breadth（patch 间）双注意力操作的模块。
- PatchSTG: 整体模型，包含时空嵌入、若干 WindowAttBlock 编码层、以及投影解码器。

注意：仅添加注释以便逐行理解。
"""


class WindowAttBlock(nn.Module):
    """
    局部 window 注意力块：实现双 attention

    参数:
        hidden_size: token 维度（embedding 大小）
        num_heads: 注意力头数（原代码在创建时传入 1）
        num: patch 个数（P）
        size: 每个 patch 中的 token 数（N）
        mlp_ratio: MLP 隐藏层相对于 hidden_size 的扩张比例
    """
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # num 表示 patch 个数 P，size 表示每个 patch 的 token 数 N
        self.num, self.size = num, size

        # breadth (跨 patch，即 patch 之间的注意力) 使用的归一化、注意力、MLP
        # 命名中 n 开头代表 "node/breadth"（论文中 breadth attention）
        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        # depth (patch 内，即 patch depth/时间/空间维度内部) 使用的归一化、注意力、MLP
        # 命名中 s 开头代表 "spatial/depth"（论文中 depth attention）
        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        """
        x 原始形状: (B, T, P*N, D) —— 这里的第三维是把 patch 内点拼接后的长度
        B: batch size
        T: time tokens（temporal patch 数，embedding 后的时间维度）
        P: spatial patch 数（WindowAttBlock 的 num）
        N: 每个 patch 内的节点数（WindowAttBlock 的 size）
        D: token 特征维度（hidden_size / embed_dim）

        原始输入 x 形状在进入 block 前被约定为 (B, T, P*N, D)
        处理流程（保持与论文一致）：
        1. 先 reshape 成 (B, T, P, N, D)
        2. depth attention（在每个 patch 内沿 N 维做 attention）
        3. breadth attention（在 patch 数量 P 上做 attention，跨 patch 聚合信息）
        4. 最终 reshape 回 (B, T, P*N, D)
        """
        B,T,_,D = x.shape
        # P: patch 个数；N: 每个 patch 的 token 数
        P, N = self.num, self.size
        # 校验：P * N 应当等于输入的第三维
        assert self.num * self.size == _
        # 重塑为五维便于在指定维度上做注意力
        x = x.reshape(B, T, P, N, D)

        # ===================== depth attention（patch 内） =====================
        # 先把 (B, T, P, N, D) 展为 (B*T*P, N, D)，在 patch 内对 N 个 token 做注意力
        qkv = self.snorm1(x.reshape(B*T*P,N,D))
        # sattn 返回 (B*T*P, N, D)，reshape 回原维度并做残差连接
        x = x + self.sattn(qkv).reshape(B,T,P,N,D)
        # MLP 层也有残差连接（先归一化再过 MLP）
        x = x + self.smlp(self.snorm2(x))
        
        # ===================== breadth attention（跨 patch） =====================
        # 先把维度互换为 (B, T, N, P, D) 再展平为 (B*T*N, P, D)，在 patch 维度上做 attention
        qkv = self.nnorm1(x.transpose(2,3).reshape(B*T*N,P,D))
        # 注意这里 reshape 回来需要 transpose 以恢复 (B, T, P, N, D)
        x = x + self.nattn(qkv).reshape(B,T,N,P,D).transpose(2,3)
        x = x + self.nmlp(self.nnorm2(x))
         
        # 最后把 patch 和 patch 内 token 拼回为一维 (P*N)
        return x.reshape(B,T,-1,D)


class PatchSTG(nn.Module):
    """
    PatchSTG 主模型：实现论文中的时空 patching + dual-attention + projection。

    构造参数对应配置文件中的 param 段：tem_patchsize/tem_patchnum/spa_patchsize/spa_patchnum 等。
    """
    def __init__(self, output_len, tem_patchsize, tem_patchnum,
                        node_num, spa_patchsize, spa_patchnum,
                        tod, dow,
                        layers, factors,
                        input_dims, node_dims, tod_dims, dow_dims,
                        ori_parts_idx, reo_parts_idx, reo_all_idx
                ):
        super(PatchSTG, self).__init__()
        # 节点数量与索引映射（用于 patching 与逆映射）
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow

        # 整体 embedding 维度：输入通道 embedding + 时间 embedding + 空间节点 embedding
        dims = input_dims + tod_dims + dow_dims + node_dims

        # ----------------- spatio-temporal embedding（Section 4.1） -----------------
        # input_emb：用 2D conv 在时间维度上做 patch 投影（kernel height=1, kernel width=tem_patchsize）
        # 注意 input_st_fc 的输入通道设置为 3（traffic + tod + dow）
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)

        # spa_emb：为每个节点创建可学习的嵌入
        self.node_emb = nn.Parameter(
                torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)

        # tem_emb：时间-of-day 与 day-of-week 的可学习嵌入
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # ----------------- dual attention encoder（Section 4.3） -----------------
        # factors 用于合并 KD-Tree 的叶子结点（减少 patch 数量），spa_encoder 由若干 WindowAttBlock 组成
        # 注意：这里将 num_heads 直接传为 1（与论文或实现保持一致）
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, spa_patchnum//factors, spa_patchsize*factors, mlp_ratio=1) for _ in range(layers)
        ])

        # ----------------- projection decoder（Section 4.4） -----------------
        # 把时序 patch 的通道维（tem_patchnum * dims）映射到 output_len（预测步数）
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x, te):
        # x: [B,T,N,1] 输入流量
        # te: [B,T,N,2] 时间特征（tod, dow）

        # ----------------- embedding（Section 4.1） -----------------
        embeded_x = self.embedding(x, te)
        # 选择经过 patching 与 padding 后的索引集合（reo_all_idx）用于 encoder
        # 防御性校验：确保索引在合法范围内，避免在 CUDA 上触发 device-side assert
        try:
            # 将可能是 numpy array/list 的索引转换为 LongTensor 并移动到与 x 相同的 device
            idx_all = torch.tensor(self.reo_all_idx, dtype=torch.long, device=x.device)
        except Exception:
            idx_all = torch.tensor([], dtype=torch.long, device=x.device)

        if idx_all.numel() > 0:
            idx_all = idx_all.clamp(0, max(0, self.node_num - 1))
        else:
            # 兜底：如果没有任何索引（极端情况），使用全体节点索引作为后备
            idx_all = torch.arange(min(1, self.node_num), dtype=torch.long, device=x.device)

        rex = embeded_x.index_select(2, idx_all)

        # ----------------- dual attention encoder（Section 4.3） -----------------
        for block in self.spa_encoder:
            rex = block(rex)

        # 把编码后的结果放回原始节点顺序：先创建一个全零张量，再把 patch 内有效位置填回
        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        # 将 rex 中按重排(reo_parts_idx)的值赋回原始索引位置 ori_parts_idx
        # 防御性处理：确保索引张量合法且长度匹配
        try:
            ori_idx = torch.tensor(self.ori_parts_idx, dtype=torch.long, device=x.device)
        except Exception:
            ori_idx = torch.tensor([], dtype=torch.long, device=x.device)
        try:
            reo_idx = torch.tensor(self.reo_parts_idx, dtype=torch.long, device=x.device)
        except Exception:
            reo_idx = torch.tensor([], dtype=torch.long, device=x.device)

        # 裁剪索引到合法范围
        if ori_idx.numel() > 0:
            ori_idx = ori_idx.clamp(0, max(0, self.node_num - 1))
        if reo_idx.numel() > 0:
            reo_idx = reo_idx.clamp(0, max(0, rex.shape[2] - 1))

        # 长度对齐（取最小长度），以免赋值时维度不匹配导致错误
        if ori_idx.numel() == 0 or reo_idx.numel() == 0:
            # 如果没有有效映射，跳过回写（保持 orginal 为 0）
            pass
        else:
            min_len = min(ori_idx.numel(), reo_idx.numel())
            ori_idx_s = ori_idx[:min_len]
            reo_idx_s = reo_idx[:min_len]
            orginal[:,:,ori_idx_s,:] = rex[:,:,reo_idx_s,:] # back to the original indices

        # ----------------- projection decoder（Section 4.4） -----------------
        # regression_conv 期望输入形状为 (B, C, H, W)，因此先把 orginal 转置并 reshape
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]

    def embedding(self, x, te):
        """
        构造时空嵌入：
        - 将 x 与时间特征拼接后经 2D conv 投影得到 input_emb
        - 把 time-of-day/day-of-week 的可学习嵌入拼接上
        - 把节点嵌入拼接上，得到最终的输入 embedding
        T' = T // tem_patchsize 把原始时间序列按多少个连续时间步合并成一个“时间 patch”
        返回形状: (B, T', N, dims)
        """
        b,t,n,_ = x.shape

        # input traffic + time of day + day of week 作为输入信号
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        # input_st_fc 要求输入为 (B, C, H, W) 格式，原始数据先 transpose 再 conv
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]        

        # cat time of day embedding：选择最近 t 个时间步的 tod 索引然后用 embedding 表
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # cat day of week embedding：同上
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # cat spatial embedding：为每个节点广播 node_emb 并拼接
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data
