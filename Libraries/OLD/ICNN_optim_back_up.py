import torch
import torch.nn as nn
import numpy as np
import random                                                                  
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic=True
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math
from torch.autograd import grad
from Libraries import robotarm
from Libraries import ICNN_net
from Libraries import ICNN_plot

dtype = torch.float
device_c = torch.device("cpu")
device = torch.device("cuda:0")

class Grid_dataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,q_in_reg):
        xy = q_in_reg #torch.from_numpy(q_in_reg)#.to(device)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.xy_data = torch.from_numpy(xy)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class ICNN_optim:
    
    def __init__(self,V_hidden_sizes, fhat_hidden_sizes,tol,alpha,Xstable,max_speed):
        super(ICNN_optim, self).__init__()
        
        self.model = ICNN_net.ICNN_net( V_hidden_sizes, fhat_hidden_sizes,tol,alpha)
        
        self.max_speed = max_speed
        self.Xstable = Xstable
    def get_grid(self,nq1,nq2,qmin,qmax):
        self.nq1 = nq1
        self.nq2 = nq2
        q_in_mesh = torch.zeros((nq1,nq2,2), device=device, dtype=dtype)
        q1_0 = (qmax[0]-qmin[0])/(2*nq1)+qmin[0]
        q2_0 = (qmax[1]-qmin[1])/(2*nq2)+qmin[1]
        q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1)
        q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2)
        q1 = torch.linspace(q1_0,q1_f,nq1)
        q2 = torch.linspace(q2_0,q2_f,nq2)
        self.dA = (q1[2]-q1[1])*(q2[2]-q2[1])
        q1_mesh,q2_mesh = torch.meshgrid(q1,q2)
        q_in_mesh[:,:,0] = q1_mesh
        q_in_mesh[:,:,1] = q2_mesh
        q_in_reg = q_in_mesh.view(-1,2).to(device_c)
        
        return q_in_reg,q1,q2
    
    def get_boundary(self,nq1_b,nq2_b,qmin,qmax):
        self.nq1_b = nq1_b
        self.nq2_b = nq2_b
        q_in_mesh = torch.zeros((nq1_b,nq2_b,2), device=device, dtype=dtype)
        q1_0 = (qmax[0]-qmin[0])/(2*nq1_b)+qmin[0]
        q2_0 = (qmax[1]-qmin[1])/(2*nq2_b)+qmin[1]
        q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1_b)
        q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2_b)
        q1 = torch.linspace(q1_0,q1_f,nq1_b)
        q2 = torch.linspace(q2_0,q2_f,nq2_b)
        q1_mesh,q2_mesh = torch.meshgrid(q1,q2)
        q_in_mesh[:,:,0] = q1_mesh
        q_in_mesh[:,:,1] = q2_mesh
        
        #getting boundary grid
        q_in_boundary = torch.zeros((nq1_b*2+nq2_b*2-4),2,device=device, dtype=dtype)
        q_in_boundary[0:nq1_b,:] = q_in_mesh[0,:,:]
        q_in_boundary[nq1_b:nq1_b+nq2_b-1,:] = q_in_mesh[1:,-1,:]
        q_in_boundary[nq1_b+nq2_b-1:nq1_b+2*nq2_b-3,:] = q_in_mesh[1:-1,0,:]
        q_in_boundary[nq1_b+2*nq2_b-3:,:] = q_in_mesh[-1,:-1,:]
        #asigning q_dot to the boundary data
        boundary_speed = 1.3*self.max_speed
        to_Xstable = self.Xstable-q_in_boundary
        norm_to_Xstable = torch.sqrt(((to_Xstable**2)[:,0] + (to_Xstable**2)[:,1])).view(-1,1)
        q_dot_boundary = (boundary_speed*(to_Xstable)/norm_to_Xstable).to(device)
        q_in_boundary.requires_grad=True
        return q_in_boundary,q_dot_boundary,q1,q2

    def get_grid2(self,nq,qmin,qmax):

        nq_torch = torch.tensor(nq)
        dim = nq_torch.shape[0]
        nq_2 = [dim]+nq
        q_in_reg = torch.zeros(nq_2, device=device, dtype=dtype)
        q0 = torch.zeros(dim,device=device, dtype=dtype)
        qf = torch.zeros(dim,device=device, dtype=dtype)
        q_total = [None] * (dim)
        
        self.V_Wx  = [None] * (V_num_hidden_layer+1)
        self.V_Wz  = [None] * (V_num_hidden_layer+1)
        for i in range(dim):
            q0[i] = (qmax[i]-qmin[i])/(2*nq[i])+qmin[i]
            qf[i] = qmax[i] - (qmax[i]-qmin[i])/(2*nq[i])
            q_total[i] = torch.linspace(q0[i],qf[i],nq[i],device=device, dtype=dtype)
            #q_in_reg[i] = 
            
            
        q1_0 = (qmax[0]-qmin[0])/(2*nq1)+qmin[0]
        q2_0 = (qmax[1]-qmin[1])/(2*nq2)+qmin[1]
        q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1)
        q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2)
        q1 = torch.linspace(q1_0,q1_f,nq1)
        q2 = torch.linspace(q2_0,q2_f,nq2)
        self.dA = (q1[2]-q1[1])*(q2[2]-q2[1])
        q1_mesh,q2_mesh = torch.meshgrid(q1,q2)
        q_in_reg[:,:,0] = q1_mesh
        q_in_reg[:,:,1] = q2_mesh
        q_in_reg = q_in_reg.view(-1,2).to(device_c)
        return q_in_reg
    
    def optim_no_reg(self,qtraj,q_dot,dt,q_in_boundary,q_dot_boundary, learning_rate = 1e-2,epoch=10000):
        #print(learning_rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        loss_mat = torch.zeros(epoch,dtype=dtype,device=device)
        for t in range(epoch):
            
            qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(device)
            qdot_pred = self.model.f_forward(q_in,self.Xstable).to(device)
            loss_traj = dt*loss_fn(qdot_pred,qdot_real).to(device)
            loss_boundary = math.pi*2/(q_in_boundary.shape[0])*loss_fn(q_dot_boundary,qdot_bound_pred).to(device)
            loss = loss_traj+loss_boundary
            loss.backward(retain_graph=True)
            optimizer.step()
            q_in.grad.zero_()
            self.model.zero_grad()
            optimizer.zero_grad()
            ld = loss.data.clone().to(device_c)
            loss_mat[t] = loss.data
            print ('\r epoch = '+str(t+1)+ ', loss = '+str(ld.data.numpy()), end = ' ')
            if t>5000:
                if (loss_mat[t]-loss_mat[t-300])/loss_mat[t] < 0.001:
                    print('Optimization finished')
                    break
        return loss_mat
        
    def optim_Euc_reg(self,q_in_reg,qtraj ,q_dot,dt,q_in_boundary,q_dot_boundary, learning_rate = 1e-2,epoch=10000,
                      batch_size = 400):
        
        dataset = Grid_dataset(q_in_reg.numpy())
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8)
        
        if batch_size>q_in_reg.shape[0]:
            print('batch_size is reduced to maximum batch_size = '+str(q_in_reg.shape[0])+'.')
        batch_size = q_in_reg.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(device)
        
        
        
        loss_fn = torch.nn.MSELoss(reduction='sum')
        loss_mat = torch.zeros(epoch,dtype=dtype,device=device)
        for t in range(epoch):
            for i, data in enumerate(train_loader):
                data0 = data[0]
                data1 = data[1]
                current_grid_data = torch.zeros(data[0].shape[0],2, device=device, dtype=dtype)
                current_grid_data[:,0]  = data[0][:,0].clone()
                current_grid_data[:,1]  = data[1][:,0].clone()
                current_grid_data.requires_grad=True
                current_output = self.model.f_forward(current_grid_data,self.Xstable).to(device)

                co1 = torch.sum(current_output[:,0])
                co2 = torch.sum(current_output[:,1])
                qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
                qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]

                qdot_grad_norm = qdot_grad1**2+qdot_grad2**2
                loss_reg = torch.sum(qdot_grad_norm)*self.dA/2*(self.nq1*self.nq2/batch_size)


                qdot_pred = self.model.f_forward(q_in,self.Xstable).to(device)
                loss_task = dt*loss_fn(qdot_pred,qdot_real).to(device)
                penalty = 200
                eps = 0.01
                loss = loss_reg+penalty*(F.relu(loss_task-eps)**2)
                #print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
                q_in.grad.zero_()
                self.model.zero_grad()
                optimizer.zero_grad()
                ld = loss.data.clone().to(device_c)
                loss_mat[t] = loss.data
                print ('\r epoch = '+str(t) +' i = '+str(i+1)+ ', loss = '
                       +str(ld.data.numpy()), end = ' ')
                if t>5000:
                    if (loss_mat[t]-loss_mat[t-300])/loss_mat[t] < 0.001:
                        print('Optimization finished')
                        return loss_mat   
        return loss_mat            
        
        
        
    
    
    
        
        