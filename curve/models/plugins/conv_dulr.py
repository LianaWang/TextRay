import torch


class ConvDU(torch.nn.Module):

    def __init__(self,
                 in_out_channels=256,
                 kernel_size=(1, 9),
                 groups=1
                 ):
        super(ConvDU, self).__init__()
        if groups > 1:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1,
                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                groups=groups),
                torch.nn.Conv2d(in_out_channels, in_out_channels, 1, stride=1, padding=(0, 0), groups=1),
                torch.nn.ReLU(inplace=True)
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1,
                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                groups=groups),
                torch.nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = [fea.select(2, i).view(n, c, 1, w) for i in range(h)]
        for i in range(1, h):
            fea_stack[i] = self.conv(fea_stack[i - 1]) + fea_stack[i]

        for i in range(h-2, 0, -1):
            fea_stack[i] = self.conv(fea_stack[i + 1]) + fea_stack[i]


        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea


class ConvLR(torch.nn.Module):

    def __init__(self,
                 in_out_channels=256,
                 kernel_size=(9, 1),
                 groups=1,
                 ):
        super(ConvLR, self).__init__()
        if groups > 1:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1,
                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                groups=groups),
                torch.nn.Conv2d(in_out_channels, in_out_channels, 1, stride=1, padding=(0, 0), groups=1),
                torch.nn.ReLU(inplace=True)
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1,
                                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                                groups=groups),
                torch.nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = [fea.select(3, i).view(n, c, h, 1) for i in range(w)]
        for i in range(1, w):
            fea_stack[i] = self.conv(fea_stack[i - 1]) + fea_stack[i]

        for i in range(w-2, 0, -1):
            fea_stack[i] = self.conv(fea_stack[i + 1]) + fea_stack[i]

        fea = torch.cat(fea_stack, 3)
        return fea
