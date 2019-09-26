import torch


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=2):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * torch.rand(1, 2, k, k, 128, requires_grad=True) - 1
        self.l1 = torch.nn.Conv3d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv3d(int(multiplier * channels), channels, 1)
        
    def forward(self, x):
        # x = self.l1(self.position)
        self.position = self.position.to(0)
        x = self.l2(torch.nn.functional.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, (self.k ** 2)*128)



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv3d(channels, channels // m, 1)
    
    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k)
        self.fold = torch.nn.Fold(output_size=(k, k), kernel_size=k)
    
    def forward(self, x):
        key_map, query_map = x
        k = self.k
        out = torch.zeros(key_map.size(0), key_map.size(1), k, k, key_map.size(4))
        for i in range(key_map.size(4)):
            key_slice = key_map[..., i]
            query_slice = query_map[..., i]
            key_map_unfold = torch.sum(self.unfold(key_slice), dim=2, keepdim=True)
            key = self.fold(key_map_unfold)
            query = query_slice[:, :, query_slice.size(2)//2, query_slice.size(3)//2]
            # query_map_unfold = torch.sum(self.unfold(query_slice), dim=2, keepdim=True)
            # query_map_fold = self.fold(query_map_unfold)
            # key_map_unfold = key_map_unfold.view(
            #             key_map.shape[0], key_map.shape[1],
            #             -1,
            #             key_map_unfold.shape[-2] // key_map.shape[1])
            # query_map_unfold = query_map_unfold.view(
            #             query_map.shape[0], query_map.shape[1],
            #             -1,
            #             query_map_unfold.shape[-2] // query_map.shape[1])
            out[..., i] = torch.einsum('bcwh,bc->bcwh', key, query)
        return out


def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel,
                                       dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, outchannels, k, stride=1, m=None, padding=0):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, m)
        self.qmap = KeyQueryMap(channels, m)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels//m)
        self.fold = torch.nn.Fold(output_size=(k, k), kernel_size=k)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv3d(m, outchannels, 1)
        
    def forward(self, x):
        km = self.kmap(x)
        qm = self.qmap(x)
        fm = torch.zeros(x.size(0), self.channels, self.k, self.k, x.size(4))
        for i in range(fm.size(4)):
            key_slice = fm[..., i]
            key_map_unfold = torch.sum(self.unfold(key_slice), dim=2, keepdim=True)
            key = self.fold(key_map_unfold)
            fm[..., i] = key
        fm = fm.view(x.size(0), self.m, self.channels//self.m, self.k, self.k, x.size(4))
        ak = self.ac((km, qm))
        ck = torch.nn.functional.softmax(ak, dim=1)
        pre_output = torch.einsum('bmcwhd,bcwhd->bmwhd', fm, ck).to(0)
        # pre_output = pre_output.view(pre_output.size(0), pre_output.size(1), 1, 1, 1)
        pre_output = torch.nn.functional.interpolate(pre_output, scale_factor=2, mode="trilinear", align_corners=False)
        output = self.final1x1(pre_output)

        return output
