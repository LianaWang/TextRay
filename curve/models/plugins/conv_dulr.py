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

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).view(n, c, 1, w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)
            # pdb.set_trace()
            # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)

        for i in range(h):
            pos = h - i - 1
            if pos == h - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
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

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).view(n, c, h, 1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)

        for i in range(w):
            pos = w - i - 1
            if pos == w - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]

        fea = torch.cat(fea_stack, 3)
        return fea
