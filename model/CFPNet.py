import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CFPNet"]

class DeConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, output_padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.ConvTranspose2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
    
    

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output



class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3,dkSize=3):
        super().__init__()
        
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)
        
        self.dconv3x1_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(1*d+1, 0), dilation=(d+1,1), groups = nIn //16, bn_acti=True)
        self.dconv1x3_4_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1*d+1), dilation=(1,d+1), groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(1*d+1, 0), dilation=(d+1,1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_4_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1*d+1), dilation=(1,d+1),groups = nIn //16, bn_acti=True)        
        
        self.dconv3x1_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(1*d+1, 0), dilation=(d+1,1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_4_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, 1*d+1), dilation=(1,d+1),groups = nIn //8, bn_acti=True) 
        
        self.dconv3x1_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(1, 0),groups = nIn //16, bn_acti=True)
        self.dconv1x3_1_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1),groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(1, 0),groups = nIn //16, bn_acti=True)
        self.dconv1x3_1_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1),groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(1, 0),groups = nIn //16, bn_acti=True)
        self.dconv1x3_1_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, 1),groups = nIn //8, bn_acti=True)
        
        
        self.dconv3x1_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/4+1), 0), dilation=(int(d/4+1),1), groups = nIn //16, bn_acti=True)
        self.dconv1x3_2_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/4+1)), dilation=(1,int(d/4+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/4+1), 0), dilation=(int(d/4+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_2_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/4+1)), dilation=(1,int(d/4+1)),groups = nIn //16, bn_acti=True)        
        
        self.dconv3x1_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(int(d/4+1), 0), dilation=(int(d/4+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_2_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, int(d/4+1)), dilation=(1,int(d/4+1)),groups = nIn //8, bn_acti=True)         
        
        
        
        self.dconv3x1_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/2+1), 0), dilation=(int(d/2+1),1), groups = nIn //16, bn_acti=True)
        self.dconv1x3_3_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/2+1)), dilation=(1,int(d/2+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/2+1), 0), dilation=(int(d/2+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_3_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/2+1)), dilation=(1,int(d/2+1)),groups = nIn //16, bn_acti=True)        
        
        self.dconv3x1_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(int(d/2+1), 0), dilation=(int(d/2+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_3_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, int(d/2+1)), dilation=(1,int(d/2+1)),groups = nIn //8, bn_acti=True)              
        
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0,bn_acti=False)
        
    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)
        
        o1_1 = self.dconv3x1_1_1(inp)
        o1_1 = self.dconv1x3_1_1(o1_1)
        o1_2 = self.dconv3x1_1_2(o1_1)
        o1_2 = self.dconv1x3_1_2(o1_2)
        o1_3 = self.dconv3x1_1_3(o1_2)
        o1_3 = self.dconv1x3_1_3(o1_3)
        
        o2_1 = self.dconv3x1_2_1(inp)
        o2_1 = self.dconv1x3_2_1(o2_1)
        o2_2 = self.dconv3x1_2_2(o2_1)
        o2_2 = self.dconv1x3_2_2(o2_2)
        o2_3 = self.dconv3x1_2_3(o2_2)
        o2_3 = self.dconv1x3_2_3(o2_3)        
     
        o3_1 = self.dconv3x1_3_1(inp)
        o3_1 = self.dconv1x3_3_1(o3_1)
        o3_2 = self.dconv3x1_3_2(o3_1)
        o3_2 = self.dconv1x3_3_2(o3_2)
        o3_3 = self.dconv3x1_3_3(o3_2)
        o3_3 = self.dconv1x3_3_3(o3_3)               
        
        
        o4_1 = self.dconv3x1_4_1(inp)
        o4_1 = self.dconv1x3_4_1(o4_1)
        o4_2 = self.dconv3x1_4_2(o4_1)
        o4_2 = self.dconv1x3_4_2(o4_2)
        o4_3 = self.dconv3x1_4_3(o4_2)
        o4_3 = self.dconv1x3_4_3(o4_3)               
        
        
        output_1 = torch.cat([o1_1,o1_2,o1_3], 1)
        output_2 = torch.cat([o2_1,o2_2,o2_3], 1)      
        output_3 = torch.cat([o3_1,o3_2,o3_3], 1)       
        output_4 = torch.cat([o4_1,o4_2,o4_3], 1)   
        
        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1,ad2,ad3,ad4],1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        
        return output+input
        

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class CFPNet(nn.Module):
    def __init__(self, classes=11, block_1=2, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)
        dilation_block_1 =[2,2]
        # CFP Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.CFP_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.CFP_Block_1.add_module("CFP_Module_1_" + str(i), CFPModule(64, d=dilation_block_1[i]))
            
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # CFP Block 2
        dilation_block_2 = [4,4,8,8,16,16] #camvid #cityscapes [4,4,8,8,16,16] # [4,8,16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.CFP_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.CFP_Block_2.add_module("CFP_Module_2_" + str(i),
                                        CFPModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)

        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # CFP Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.CFP_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # CFP Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.CFP_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out
    
    
