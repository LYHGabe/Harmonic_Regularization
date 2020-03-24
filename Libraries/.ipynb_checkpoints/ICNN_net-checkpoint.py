import torch
import torch.nn as nn
import numpy as np
import random             
from torch.autograd import Variable
torch.backends.cudnn.deterministic=True
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd import grad
import math

dtype = torch.float
device_c = torch.device("cpu")
device = torch.device("cuda:0")

class Smooth_ReLU(torch.autograd.Function):
    

    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        output = torch.zeros(input.shape,dtype=dtype)#,device=device)
        output = input.float().clone()-1/2
        output[input < 1] = (input.float()[input < 1] ** 2)/(2)
        output[input < 0 ] = 0
        return output

    @staticmethod
    def backward(ctx,grad_output):
        d=1
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[input < 0] = 0
        grad[input >= 0] = input[input >= 0]
        grad[input > 1] = 1
        grad = torch.mul(grad,grad_output)
        return grad
    
class Weightsum_ICNN(nn.Module):

    def __init__(self, input_dim, output_dim, bias=True):
        super(Weightsum_ICNN, self).__init__()
        self.icnnx = nn.Linear(2, output_dim, bias=False)    
        self.icnnz = nn.Linear(input_dim, output_dim, bias=False)        
        torch.nn.init.xavier_normal_(self.icnnx.weight)     
        torch.nn.init.xavier_normal_(self.icnnz.weight)
        #torch.nn.init.uniform_(self.icnnx.weight, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.icnnx.weight, a=0.0, b=1.0)
    def forward(self, X, z):
        
        out = self.icnnx(X) + self.icnnz(z)
        
        return out

    
class ICNN_net(nn.Module):
    
    def __init__(self, V_hidden_sizes, fhat_hidden_sizes, tol, alpha,device, slope = 0.01):
        super(ICNN_net, self).__init__()
        self.V_inputSize = 2
        self.V_outputSize = 1
        self.fhat_inputSize = 2
        self.fhat_outputSize = 2
        
        self.ReLU = nn.ReLU()
        self.Smooth_ReLU = Smooth_ReLU.apply
        self.fhat_activation = nn.LeakyReLU(negative_slope = slope)
        self.Sigmoid = nn.Sigmoid()
        self.tol = tol
        self.alpha = alpha
        self.device = device
        
        self.Vl1 = nn.Linear(self.V_inputSize,V_hidden_sizes[0],bias=False).to(self.device)
        torch.nn.init.xavier_normal_(self.Vl1.weight)
        self.Vl2 = Weightsum_ICNN(V_hidden_sizes[0],V_hidden_sizes[1]).to(self.device)
        self.Vl3 = Weightsum_ICNN(V_hidden_sizes[1],V_hidden_sizes[2]).to(self.device)
        #self.Vl4 = Weightsum_ICNN(V_hidden_sizes[2],V_hidden_sizes[3]).to(self.device)
        self.Vlf = Weightsum_ICNN(V_hidden_sizes[-1],self.V_outputSize).to(self.device)
        
        self.fhatl1 = nn.Linear(self.fhat_inputSize,fhat_hidden_sizes[0],bias=True).to(self.device)
        self.fhatl2 = nn.Linear(fhat_hidden_sizes[0],fhat_hidden_sizes[1],bias=True).to(self.device)
        self.fhatl3 = nn.Linear(fhat_hidden_sizes[1],fhat_hidden_sizes[2],bias=True).to(self.device)
        self.fhatl4 = nn.Linear(fhat_hidden_sizes[2],fhat_hidden_sizes[3],bias=True).to(self.device)
        self.fhatl5 = nn.Linear(fhat_hidden_sizes[3],fhat_hidden_sizes[4],bias=True).to(self.device)
        #self.fhatl6 = nn.Linear(fhat_hidden_sizes[4],fhat_hidden_sizes[3],bias=True).to(self.device)
        #self.fhatl7 = nn.Linear(fhat_hidden_sizes[5],fhat_hidden_sizes[4],bias=True).to(self.device)
        #self.fhatl8 = nn.Linear(fhat_hidden_sizes[6],fhat_hidden_sizes[3],bias=True).to(self.device)
        #self.fhatl9 = nn.Linear(fhat_hidden_sizes[7],fhat_hidden_sizes[4],bias=True).to(self.device)
        self.fhatlf = nn.Linear(fhat_hidden_sizes[-1],self.fhat_outputSize,bias=True).to(self.device)
        #torch.nn.init.uniform_(self.fhatl1, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl2, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl3, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl4, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl5, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl6, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl7, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl8, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatl9, a=0.0, b=1.0)
        #torch.nn.init.uniform_(self.fhatlf, a=0.0, b=1.0)
        
        torch.nn.init.xavier_normal_(self.fhatl1.weight)
        torch.nn.init.xavier_normal_(self.fhatl2.weight)
        torch.nn.init.xavier_normal_(self.fhatl3.weight)
        torch.nn.init.xavier_normal_(self.fhatl4.weight)
        torch.nn.init.xavier_normal_(self.fhatl5.weight)
        #torch.nn.init.xavier_normal_(self.fhatl6.weight)
        #torch.nn.init.xavier_normal_(self.fhatl7.weight)
        #torch.nn.init.xavier_normal_(self.fhatl8.weight)
        #torch.nn.init.xavier_normal_(self.fhatl9.weight)
        torch.nn.init.xavier_normal_(self.fhatlf.weight)
        
    def V_forward(self, X, Xstable):
        
        V_z1_pre = self.Vl1(X-Xstable)
        V_z1 = self.Smooth_ReLU(V_z1_pre)
        V_z2_pre = self.Vl2(X-Xstable,V_z1)
        V_z2 = self.Smooth_ReLU(V_z2_pre)
        V_z3_pre = self.Vl3(X-Xstable,V_z2)
        V_z3 = self.Smooth_ReLU(V_z3_pre)
        #V_z4_pre = self.Vl4(X-Xstable,V_z3)
        #V_z4 = self.Smooth_ReLU(V_z4_pre)
        V_zf_pre = self.Vlf(X-Xstable,V_z3)
        V_zf = self.Smooth_ReLU(V_zf_pre)
        
        Vsum_pre = self.tol*((X-Xstable) ** 2)
        Xst = torch.zeros(X.shape).to(self.device)
        Vsum = torch.zeros(X.shape[0],1).to(self.device)
        Vsum[:,0] = (Vsum_pre[:,0]+Vsum_pre[:,1])
        V = self.Smooth_ReLU(V_zf) + Vsum
        return V
    
    
    def fhat_forward(self, X):
        fhat_z1_pre = self.fhatl1(X)
        fhat_z1 = self.fhat_activation(fhat_z1_pre)
        fhat_z2_pre = self.fhatl2(fhat_z1)
        fhat_z2 = self.fhat_activation(fhat_z2_pre)
        fhat_z3_pre = self.fhatl3(fhat_z2)
        fhat_z3 = self.fhat_activation(fhat_z3_pre)
        fhat_z4_pre = self.fhatl4(fhat_z3)
        fhat_z4 = self.fhat_activation(fhat_z4_pre)
        fhat_z5_pre = self.fhatl5(fhat_z4)
        fhat_z5 = self.fhat_activation(fhat_z5_pre)
        #fhat_z6_pre = self.fhatl5(fhat_z5)
        #fhat_z6 = self.fhat_activation(fhat_z6_pre)
        #fhat_z7_pre = self.fhatl5(fhat_z6)
        #fhat_z7 = self.fhat_activation(fhat_z7_pre)
        #fhat_z8_pre = self.fhatl5(fhat_z7)
        #fhat_z8 = self.fhat_activation(fhat_z8_pre)
        #fhat_z9_pre = self.fhatl5(fhat_z8)
        #fhat_z9 = self.fhat_activation(fhat_z9_pre)
        fhat_zf = self.fhatlf(fhat_z5)
        
        fhat = fhat_zf
        
        return fhat
    
    def f_forward(self, X, Xstable):
        V = self.V_forward(X, Xstable)
        f_hat = self.fhat_forward(X)
        f = torch.zeros(f_hat.shape)
        

        Vsum = torch.sum(torch.sum(V))
        gradV = grad(Vsum,X,create_graph=True,retain_graph=True)[0]
        Vnorm0 = gradV ** 2
        Vnorm = torch.zeros(V.shape,dtype=dtype,device=self.device)
        Vnorm[:,0] = Vnorm0[:,0]+Vnorm0[:,1]
        fh_V_mul0 = torch.mul(f_hat,gradV)
        fh_V_mul = torch.zeros(V.shape,dtype=dtype,device=self.device)
        fh_V_mul[:,0] = fh_V_mul0[:,0]+fh_V_mul0[:,1]
        eps = 1.0e-10
        f_mul = F.relu((fh_V_mul+self.alpha*V))/(Vnorm+eps)
        f = f_hat - gradV*f_mul
        #f = f_hat
        
        
        return f