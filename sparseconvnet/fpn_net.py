# xyz Jan 2019

import torch
import torch.nn as nn
import sparseconvnet as scn
from .sparseConvNetTensor import SparseConvNetTensor

def sparse_info(t):
  print(f'{t.features.shape}, {t.spatial_size}')

class FPN_Net(torch.nn.Module):
    _show = False
    def __init__(self, full_scale, dimension, reps, nPlanes, residual_blocks=False,
                  downsample=[2, 2], leakiness=0):
        '''
        downsample:[kernel, stride]
        '''
        nn.Module.__init__(self)

        self._merge = 'cat'  # 'cat' or 'add'

        self.layers_in = scn.Sequential(
                scn.InputLayer(dimension,full_scale, mode=4),
                scn.SubmanifoldConvolution(dimension, 3, nPlanes[0], 3, False))

        self.layers_out = scn.Sequential(
            scn.BatchNormReLU(nPlanes[0]),
            scn.OutputLayer(dimension))

        self.linear = nn.Linear(nPlanes[0], 20)

        #**********************************************************************#

        def block(m, a, b):
            if residual_blocks: #ResNet style blocks
                m.add(scn.ConcatTable()
                      .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                      .add(scn.Sequential()
                        .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                        .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                        .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                        .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
                 ).add(scn.AddTable())
            else: #VGG style blocks
                m.add(scn.Sequential()
                     .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                     .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))

        def down(m, nPlane_in, nPlane_downed):
            m.add(scn.Sequential()
                  .add(scn.BatchNormLeakyReLU(nPlane_in,leakiness=leakiness))
                  .add(scn.Convolution(dimension, nPlane_in, nPlane_downed,
                          downsample[0], downsample[1], False)))

        def up(m, nPlane_in, nPlane_uped):
           m.add( scn.BatchNormLeakyReLU(nPlane_in, leakiness=leakiness)).add(
                      scn.Deconvolution(dimension, nPlane_in, nPlane_uped,
                      downsample[0], downsample[1], False))


        scales_num = len(nPlanes)
        m_downs = nn.ModuleList()
        for k in range(scales_num):
            m = scn.Sequential()
            if k > 0:
              down(m, nPlanes[k-1], nPlanes[k])
            for _ in range(reps):
                block(m, nPlanes[k], nPlanes[k])
            m_downs.append(m)

        m_ups = nn.ModuleList()
        m_ups_decoder = nn.ModuleList()
        for k in range(scales_num-1, 0, -1):
            m = scn.Sequential()
            up(m, nPlanes[k], nPlanes[k-1])
            m_ups.append(m)

            m = scn.Sequential()
            for i in range(reps):
                block(m, nPlanes[k-1] * (1+int(self._merge=='cat') if i == 0 else 1), nPlanes[k-1])
            m_ups_decoder.append(m)

        self.m_downs = m_downs
        self.m_ups = m_ups
        self.m_ups_decoder = m_ups_decoder

    def forward(self, net0):
      if self._show: print(f'\ninput: {net0[0].shape}')
      net1 = self.layers_in(net0)
      net_scales = self.forward_fpn(net1)
      net = net_scales[-1]
      net = self.layers_out(net)
      net = self.linear(net)
      if self._show:
        print(f'\nend {net.shape}\n')
      return net

    def forward_fpn(self, net):
      scales_num = len(self.m_downs)
      downs = []
      for m in self.m_downs:
        net = m(net)
        downs.append(net)

      ups = [net]
      for k in range(scales_num-1):
        j = scales_num-1-k-1
        net = self.m_ups[k](net)
        if self._merge == 'add':
          net = scn.add_feature_planes([net, downs[j]])
        if self._merge == 'cat':
          net = scn.concatenate_feature_planes([net, downs[j]])
        net = self.m_ups_decoder[k](net)
        ups.append(net)

      if self._show:
        print('\ndowns:')
        [sparse_info(t) for t in downs]
        print('\nups:')
        [sparse_info(t) for t in ups]
      return ups

