import torch


class ResBlocks(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        n_convs: int,
        # attentions: int,
    ) -> None:
        super().__init__()
        self.factor_skip_conv = 0.5
        self.factor_skip_res = 0.5
        num_groups = 4
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        dilation_base = 2
        conv_groups = 1  # TODO: experiment with wide kernels, dilation base > 2 and conv groups > 1
        self.activation = torch.nn.GELU()
        self.norms = torch.nn.ModuleList(
            torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            for _ in range(n_convs)
        )
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding * dilation_base**i,
                dilation=dilation_base**i,
                groups=conv_groups,
            )
            for i in range(n_convs)
        )
    
    def forward(self, x):
        x_res = x
        for norm, conv in zip(self.norms, self.convs):
            x_post_res = norm(x_res)
            x_post_res = conv(x_post_res)
            x_post_res = self.activation(x_post_res)
            x_res = self.factor_skip_conv * x_res + (1 - self.factor_skip_conv) * x_post_res
        return self.factor_skip_res * x + (1 - self.factor_skip_res) * x_res


class SkipCat(torch.nn.Module):
    def __init__(
        self,
        stack: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.stack = stack
    
    def forward(self, x: torch.Tensor):
        y = self.stack(x)
        return torch.cat((x, y), dim=1)




class UNetLayer(torch.nn.Module):
    def __init__(
        self,
        channels: list[int],
        factors: list[int],
        blocks: list[int],  # TODO: why not different amount of blocks for up / down path?
        # attentions: list[int],
    ) -> None:
        super().__init__()
        # recursive stuff
        assert len(channels) == len(factors) + 2
        assert len(channels) == len(blocks) + 1
        c1, c2 = channels[:2]
        in_conv = torch.nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        out_conv = torch.nn.Conv1d(c2, c1, kernel_size=1)
        in_res = ResBlocks(c2, blocks[0])
        out_res = ResBlocks(c2, blocks[0])
        # self.skip_conv = torch.nn.Conv1d(c2 * 2, c2, kernel_size=1)
        if len(factors) > 0:
            sample_down = torch.nn.MaxPool1d(kernel_size=factors[0], stride=factors[0])  # TODO: experiment with different sampling methods
            sample_up = torch.nn.Upsample(scale_factor=factors[0])

            inner = UNetLayer(
                channels=channels[1:],
                factors=factors[1:],
                blocks=blocks[1:],
            )
            inner = torch.nn.Sequential(
                sample_down,
                inner,
                sample_up
            )
            scale = torch.nn.Sequential(
                SkipCat(inner),
                torch.nn.Conv1d(c2 * 2, c2, kernel_size=1),
            )
        else:
            scale = torch.nn.Identity()
        
        self.layer_stack = torch.nn.Sequential(
            in_conv,
            torch.nn.GELU(),
            in_res,
            scale,
            out_res,
            out_conv,
            torch.nn.GELU(),
        )
        # self.forward = self.layer_stack.forward  # TODO: skipping hooks?
        
    def forward(self, x):
        return self.layer_stack(x)


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        factors: list[int],
        blocks: list[int],
        # attentions: list[int],
    ) -> None:
        super().__init__()
        conv_in = torch.nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)
        conv_out = torch.nn.Conv1d(channels[0], out_channels, kernel_size=1)
        self.stack = torch.nn.Sequential(
            conv_in,
            torch.nn.GELU(),
            UNetLayer(channels, factors, blocks),
            conv_out,
        )
    
    def forward(self, x, *args, **kwargs):
        # TODO: do something about the time embeddings
        return self.stack(x)
        