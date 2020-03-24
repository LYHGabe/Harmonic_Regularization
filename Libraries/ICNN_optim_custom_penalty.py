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
    
    def __init__(self,V_hidden_sizes, fhat_hidden_sizes,tol,alpha,Xstable,device,T,slope = 0.01):
        super(ICNN_optim, self).__init__()
        self.device = device
        self.model = ICNN_net.ICNN_net( V_hidden_sizes, fhat_hidden_sizes,tol,alpha,self.device,slope = slope)
        self.Xstable = Xstable
        self.expmul = 2
        self.n_totalgrid = 10
        self.T = T
        
       
    def penalty_fun(self,loss,eps,epoch,mode = 0):
        if mode==0:
            return torch.exp(self.expmul*F.relu(loss-eps))
        if mode==1:
            return F.relu(loss-eps)**2
        if mode==2:
            t= 2
            return -1/t*torch.log(-(loss-eps))
    
    
    
    
    
    def get_grid(self,nq1,nq2,qmin,qmax):
        self.nq1 = nq1
        self.nq2 = nq2
        self.qmin = qmin
        self.qmax = qmax
        q_in_mesh = torch.zeros((nq1,nq2,2), device=self.device, dtype=dtype)
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
    
    
    
    def get_qsmall(self,nq1,nq2,qmin,qmax):
        #self.nq1 = nq1
        #self.nq2 = nq2
        #self.qmin = qmin
        #self.qmax = qmax
        q_in_mesh = torch.zeros((nq1,nq2,2), device=self.device, dtype=dtype)
        q1_0 = (qmax[0]-qmin[0])/(2*nq1)+qmin[0]
        q2_0 = (qmax[1]-qmin[1])/(2*nq2)+qmin[1]
        q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1)
        q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2)
        q1 = torch.linspace(q1_0,q1_f,nq1)
        q2 = torch.linspace(q2_0,q2_f,nq2)
        #self.dA = (q1[2]-q1[1])*(q2[2]-q2[1])
        q1_mesh,q2_mesh = torch.meshgrid(q1,q2)
        q_in_mesh[:,:,0] = q1_mesh
        q_in_mesh[:,:,1] = q2_mesh
        q_in_reg = q_in_mesh.view(-1,2).to(device_c)
        
        return q_in_reg
    
    
    def get_boundary(self,nq1_b,nq2_b,qmin,qmax):
        self.nq1_b = nq1_b
        self.nq2_b = nq2_b
        q_in_mesh = torch.zeros((nq1_b,nq2_b,2), device=self.device, dtype=dtype)
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
        q_in_boundary = torch.zeros((nq1_b*2+nq2_b*2),2,device=self.device, dtype=dtype)
        q_in_boundary[0:nq1_b,:] = q_in_mesh[0,:,:]
        q_in_boundary[0:nq1_b,0] = qmin[0]
        q_in_boundary[nq1_b:nq1_b+nq2_b,:] = q_in_mesh[:,-1,:]
        q_in_boundary[nq1_b:nq1_b+nq2_b,1] = qmax[1]
        q_in_boundary[nq1_b+nq2_b:nq1_b+2*nq2_b,:] = q_in_mesh[:,0,:]
        q_in_boundary[nq1_b+nq2_b:nq1_b+2*nq2_b,1] = qmin[1]
        q_in_boundary[nq1_b+2*nq2_b:,:] = q_in_mesh[-1,:,:]
        q_in_boundary[nq1_b+2*nq2_b:,0] = qmax[0]
        
        
        #asigning q_dot to the boundary data
        boundary_speed = 1.0
        q_dot_boundary  = torch.zeros((nq1_b*2+nq2_b*2),2,device=self.device, dtype=dtype)
        q_dot_boundary[0:nq1_b,0] = 1
        q_dot_boundary[nq1_b:nq1_b+nq2_b,1] = -1
        q_dot_boundary[nq1_b+nq2_b:nq1_b+2*nq2_b,1] = 1
        q_dot_boundary[nq1_b+2*nq2_b:,0] = -1
        
        q_in_boundary.requires_grad=True
        return q_in_boundary,q_dot_boundary

    def get_grid2(self,nq,qmin,qmax): #not using it yet..

        nq_torch = torch.tensor(nq)
        dim = nq_torch.shape[0]
        nq_2 = [dim]+nq
        q_in_reg = torch.zeros(nq_2, device=self.device, dtype=dtype)
        q0 = torch.zeros(dim,device=self.device, dtype=dtype)
        qf = torch.zeros(dim,device=self.device, dtype=dtype)
        q_total = [None] * (dim)
        
        self.V_Wx  = [None] * (V_num_hidden_layer+1)
        self.V_Wz  = [None] * (V_num_hidden_layer+1)
        for i in range(dim):
            q0[i] = (qmax[i]-qmin[i])/(2*nq[i])+qmin[i]
            qf[i] = qmax[i] - (qmax[i]-qmin[i])/(2*nq[i])
            q_total[i] = torch.linspace(q0[i],qf[i],nq[i],device=self.device, dtype=dtype)
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
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        #loss_mat = torch.zeros(epoch,dtype=dtype,device=self.device)
        for t in range(epoch):
            
            qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
            temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
            temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
            temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
            loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
            penalty_boundary = 1
            
            
            qdot_pred = self.model.f_forward(q_in,self.Xstable).to(self.device)
            loss_traj = dt*loss_fn(qdot_pred,qdot_real).to(self.device)
            
            loss = loss_traj+penalty_boundary*((loss_boundary)**2)
            loss.backward(retain_graph=True)
            optimizer.step()
            q_in.grad.zero_()
            q_in_boundary.grad.zero_()
            self.model.zero_grad()
            optimizer.zero_grad()
            ld = loss.data.clone().to(device_c)
            #loss_mat[t] = loss_boundary.data
            print ('\r epoch = '+str(t+1)+ ', loss = '
                       +str(ld.data.numpy()) +', loss_b = '+str(loss_boundary.to(device_c).data.numpy())+ 
                   ', loss_t = '+str(loss_traj.to(device_c).data.numpy())+'                    .', end = ' ')
            
            if t>50000:
                if (loss_mat[t]-loss_mat[t-1000])/loss_mat[t] < 0.001:
                    print('Optimization finished')
                    break
                    
    def optim_weight_reg(self,qtraj,q_dot,dt,q_in_boundary,q_dot_boundary, learning_rate = 1e-2,epoch=10000,weight_decay = 1e-3):
        #print(learning_rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,weight_decay=weight_decay)
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        #loss_mat = torch.zeros(epoch,dtype=dtype,device=self.device)
        for t in range(epoch):
            
            qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
            temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
            temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
            temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
            loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
            penalty_boundary = 1
            
            
            qdot_pred = self.model.f_forward(q_in,self.Xstable).to(self.device)
            loss_traj = dt*loss_fn(qdot_pred,qdot_real).to(self.device)
            
            loss = loss_traj+penalty_boundary*((loss_boundary)**2)
            loss.backward(retain_graph=True)
            optimizer.step()
            q_in.grad.zero_()
            q_in_boundary.grad.zero_()
            self.model.zero_grad()
            optimizer.zero_grad()
            ld = loss.data.clone().to(device_c)
            #loss_mat[t] = loss_boundary.data
            print ('\r epoch = '+str(t+1)+ ', loss = '
                       +str(ld.data.numpy()) +', loss_b = '+str(loss_boundary.to(device_c).data.numpy())+ 
                   ', loss_t = '+str(loss_traj.to(device_c).data.numpy())+'                    .', end = ' ')
            
            if t>50000:
                if (loss_mat[t]-loss_mat[t-1000])/loss_mat[t] < 0.001:
                    print('Optimization finished')
                    break
                    
                    
    def optim_traj_reg(self,qtraj,q_dot,dt,q_in_boundary,q_dot_boundary, learning_rate = 1e-2,epoch=10000):
        
        #print(qtraj.shape)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        current_grid_data = q_in
        #current_grid_data.requires_grad=True
        #loss_mat = torch.zeros(epoch,dtype=dtype,device=self.device)
        
        for t in range(epoch):
            
            qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
            temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
            temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
            temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
            loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
            penalty_boundary = 1
            
            
            qdot_pred = self.model.f_forward(q_in,self.Xstable).to(self.device)
            current_output = self.model.f_forward(current_grid_data,self.Xstable).to(self.device)

            co1 = torch.sum(current_output[:,0])
            co2 = torch.sum(current_output[:,1])
            
            qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
            qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]

            qdot_grad_norm = qdot_grad1**2+qdot_grad2**2
            loss_reg = torch.sum(qdot_grad_norm)*dt*0.01 #dx/dt 구해서 넣는거 추가해야함~!
                
            loss_traj = dt*loss_fn(qdot_pred,qdot_real).to(self.device)
            
            loss = loss_traj+penalty_boundary*((loss_boundary)**2)+loss_reg
            loss.backward(retain_graph=True)
            optimizer.step()
            q_in.grad.zero_()
            q_in_boundary.grad.zero_()
            self.model.zero_grad()
            optimizer.zero_grad()
            ld = loss.data.clone().to(device_c)
            #loss_mat[t] = loss_boundary.data
            print ('\r epoch = '+str(t) + 
                       ', loss = ' +str(ld.data.numpy())+
                       ', loss_b = '+str(loss_boundary.to(device_c).data.numpy())+ 
                       ', loss_t = '+str(loss_traj.to(device_c).data.numpy())+
                       ', loss_reg = '+str(loss_reg.to(device_c).data.numpy())+'           .',end = ' ')
            
            if t>50000:
                if (loss_mat[t]-loss_mat[t-1000])/loss_mat[t] < 0.001:
                    print('Optimization finished')
                    break
                    
        
    
     
    def optim_Euc_reg(self,q_in_reg,qtraj ,q_dot,dt,q_in_boundary,q_dot_boundary,penalty,penalty_boundary, learning_rate = 1e-2,epoch=10000,
                      batch_size = 400,penalty_mode = 0):
        
        dataset = Grid_dataset(q_in_reg.numpy())
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8)
        
        
        if batch_size>q_in_reg.shape[0]:
            #print('batch_size is reduced to maximum batch_size = '+str(q_in_reg.shape[0])+'.')
            batch_size = q_in_reg.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        
        
        
        loss_fn = torch.nn.MSELoss(reduction='sum')
        #loss_mat = torch.zeros(epoch,dtype=dtype,device=self.device)
        for t in range(epoch):
            for i, data in enumerate(train_loader):
                
                qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
                temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
                temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
                temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
                
                loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
                #penalty_boundary = 10000                             ##########################################################################
                
                
                
                data0 = data[0]
                data1 = data[1]
                current_grid_data = torch.zeros(data[0].shape[0],2, device=self.device, dtype=dtype)
                current_grid_data[:,0]  = data[0][:,0].clone()
                current_grid_data[:,1]  = data[1][:,0].clone()
                current_grid_data.requires_grad=True
                
                current_output = self.model.f_forward(current_grid_data,self.Xstable).to(self.device)

                co1 = torch.sum(current_output[:,0])
                co2 = torch.sum(current_output[:,1])
                
                qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
                qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]

                qdot_grad_norm = qdot_grad1**2+qdot_grad2**2
                #print(self.nq1*self.nq2, q_in_reg.shape[0])
                loss_reg = torch.sum(qdot_grad_norm)*self.dA/2*(self.nq1*self.nq2/batch_size)

                qdot_pred = self.model.f_forward(q_in,self.Xstable).to(self.device)
                loss_task = dt*loss_fn(qdot_pred,qdot_real).to(self.device)
                ##################
                #fsum = torch.sum(qdot_bound_pred**2)-1
                
                
                #penalty = 10000                             ##########################################################################
                eps = 0.1
                loss = loss_reg+penalty*(self.penalty_fun(loss_task,eps,epoch,mode = penalty_mode))+penalty_boundary*((loss_boundary)**2)
                
                #print(loss)
                loss.backward(retain_graph=True)
                max_grad_norm = 1
                norm_type=2
                if max_grad_norm > 0:
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
                
                optimizer.step()
                q_in.grad.zero_()
                self.model.zero_grad()
                optimizer.zero_grad()
                current_grid_data.grad.zero_()
                ld = loss.data.clone().to(device_c)
                #loss_mat[t] = loss.data
                #del loss
                #del co1
                #del co2
                #del qdot_grad1
                #del qdot_grad2
                #del current_output
                #del current_grid_data
                #del qdot_grad_norm
                
                
                if i%10==0:
                    total_grid = q_in_reg.to(self.device)
                    total_grid.requires_grad=True
                    current_output = self.model.f_forward(total_grid,self.Xstable).to(self.device)
                
                    co1 = torch.sum(current_output[:,0])
                    co2 = torch.sum(current_output[:,1])
                
                    qdot_grad1 = grad(co1,total_grid,retain_graph=True)[0]
                    qdot_grad2 = grad(co2,total_grid,retain_graph=True)[0]
                    
                    qdot_grad_norm = qdot_grad1**2+qdot_grad2**2
                    total_loss_reg = torch.sum(qdot_grad_norm)*self.dA/2
                    total_loss = (total_loss_reg+penalty*(self.penalty_fun(loss_task,eps,epoch,mode = penalty_mode))+penalty_boundary*((loss_boundary)**2)).data.to(device_c)
                    total_loss_reg = total_loss_reg.data.to(device_c)
                    
                    #del co1
                    #del co2
                    #del qdot_grad1
                    #del qdot_grad2
                    #del current_output
                    #del total_grid
                    #del qdot_grad_norm
                #total_loss = ld.data
                #total_loss_reg = loss_reg.data
                print ('\r epoch = '+str(t) +' i = '+str(i+1)+ 
                       ', loss = ' +str(ld.data.numpy())+
                       ', total_loss = ' +str(total_loss.data.numpy())+
                       ', loss_b = '+str(loss_boundary.to(device_c).data.numpy())+ 
                       ', loss_t = '+str(loss_task.to(device_c).data.numpy())+
                       ', loss_reg = '+str(loss_reg.to(device_c).data.numpy())+
                       ', loss_reg_total =  '+str(total_loss_reg.to(device_c).data.numpy())+'           .',end = ' ')
                
                if t>10000:
                    if (loss_mat[t]-loss_mat[t-300])/loss_mat[t] < 0.001:
                        print('Optimization finished')
        
        
    def optim_Kinematic_reg(self, robot, q_in_reg, qtraj, q_dot, dt,q_in_boundary,q_dot_boundary,penalty,penalty_boundary, Xstable,alpha_kinematic, learning_rate = 1e-3, epoch=10000,
                      batch_size = 400,penalty_mode = 0):
        
        
        
        q_in_small = self.get_qsmall(self.n_totalgrid,self.n_totalgrid,self.qmin,self.qmax)
        q_in_small = q_in_small.to(self.device)
        q_in_small.requires_grad = True
        
        dataset = Grid_dataset(q_in_reg.numpy())
        train_loader = DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)
        if batch_size>q_in_reg.shape[0]:
            #print('batch_size is reduced to maximum batch_size = '+str(q_in_reg.shape[0])+'.')
            batch_size = q_in_reg.shape[0]
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        #loss_mat = torch.zeros(epoch,dtype=dtype,device=self.device)

        for t in range(epoch):
            for i, data in enumerate(train_loader):
                
                qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
                temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
                temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
                temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
                
                loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
                #penalty_boundary = 10000                             ##########################################################################

                data0 = data[0]
                data1 = data[1]
                current_grid_data = torch.zeros(data[0].shape[0],2, device=self.device, dtype=dtype)
                current_grid_data[:,0]  = data[0][:,0].clone()
                current_grid_data[:,1]  = data[1][:,0].clone()
                current_grid_data.requires_grad=True

                current_output = self.model.f_forward(current_grid_data,Xstable).to(self.device)
                current_output = current_output.t()
                co1 = torch.sum(current_output[:,0])
                co2 = torch.sum(current_output[:,1])
                qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
                qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]
                qdot_grad_together = torch.zeros(2,2,current_grid_data.shape[0]).to(self.device)
                qdot_grad_together[:,0,:] = qdot_grad1.t()
                qdot_grad_together[:,1,:] = qdot_grad2.t()

                ####################################### 
                num_sampled_batch = current_grid_data.shape[0]
                Matrix_del_f = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_Gamma = torch.zeros(2*num_sampled_batch,2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_f = torch.zeros(2*num_sampled_batch,1).to(self.device)
                Matrix_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_det_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_G_inv = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                for k in range(num_sampled_batch):
                    Temp1 = robot.get_Christoffel_kinematic(current_grid_data[k,:],alpha_kinematic)
                    Temp2 = robot.get_Kinematic_Riemannian_metric(current_grid_data[k,:],alpha_kinematic)

                    Matrix_del_f[2*k:2*(k+1),2*k:2*(k+1)] = qdot_grad_together[:,:,k]
                    Gamma = torch.zeros(2,2,2)
                    Gamma[:,:,0] = Temp1[0].detach().to(self.device)
                    Gamma[:,:,1] = Temp1[1].detach().to(self.device)
                    Matrix_Gamma[2*k:2*(k+1),2*k:2*(k+1),2*k:2*(k+1)] = Gamma
                    Matrix_f[2*k:2*(k+1),:] = torch.reshape(current_output[:,k],[2,1])
                    Matrix_G[2*k:2*(k+1),2*k:2*(k+1)] = Temp2
                    Matrix_det_G[2*k:2*(k+1),2*k:2*(k+1)] = torch.sqrt(torch.det(Temp2))*torch.eye(2)
                    Matrix_G_inv[2*k:2*(k+1),2*k:2*(k+1)] = torch.inverse(Temp2)
                ######################################
                
                cov_der = Matrix_del_f+torch.tensordot(Matrix_Gamma,Matrix_f,dims=([2],[0])).reshape(2*num_sampled_batch,2*num_sampled_batch)
                #print(cov_der)
                Integral_approximation = torch.trace(torch.mm(torch.mm(torch.mm(torch.mm(cov_der.t(),Matrix_G),cov_der),Matrix_G_inv),Matrix_det_G))
                loss_reg = Integral_approximation*self.dA/2*(self.nq1*self.nq2/batch_size)
                
                qdot_pred = self.model.f_forward(q_in,Xstable).to(self.device)
                loss_task = dt*loss_fn(qdot_pred,qdot_real).to(self.device)
                #penalty = 10000                             ##########################################################################
                eps = 0.1
                loss = loss_reg+penalty*(self.penalty_fun(loss_task,eps,epoch,mode = penalty_mode))+penalty_boundary*((loss_boundary)**2)
                loss.backward(retain_graph=True)
                max_grad_norm = 1
                norm_type=2
            
                if max_grad_norm > 0:
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
                        
                optimizer.step()
                q_in.grad.zero_()
                self.model.zero_grad()
                optimizer.zero_grad()
                current_grid_data.grad.zero_()
                ld = loss.data.clone().to(device_c)
                #loss_mat[t] = loss.data
                ###############################################    ##########################              #########################
                if i%10==0:
                    
                    #total_grid = q_in_small.to(self.device)
                    #total_grid.requires_grad=True
                    total_grid = q_in_small
                    current_output = self.model.f_forward(total_grid,self.Xstable).to(self.device)
                    current_output = current_output.t()
                    co1 = torch.sum(current_output[:,0])
                    co2 = torch.sum(current_output[:,1])
                    qdot_grad1 = grad(co1,total_grid,retain_graph=True)[0]
                    qdot_grad2 = grad(co2,total_grid,retain_graph=True)[0]
                    qdot_grad_together = torch.zeros(2,2,total_grid.shape[0]).to(self.device)
                    qdot_grad_together[:,0,:] = qdot_grad1.t()
                    qdot_grad_together[:,1,:] = qdot_grad2.t()

                    ####################################### 
                    num_sampled_batch = total_grid.shape[0]
                    Matrix_del_f = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_Gamma = torch.zeros(2*num_sampled_batch,2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_f = torch.zeros(2*num_sampled_batch,1).to(self.device)
                    Matrix_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_det_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_G_inv = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    for k in range(num_sampled_batch):
                        Temp1 = robot.get_Christoffel_kinematic(total_grid[k,:],alpha_kinematic)
                        Temp2 = robot.get_Kinematic_Riemannian_metric(total_grid[k,:],alpha_kinematic)

                        Matrix_del_f[2*k:2*(k+1),2*k:2*(k+1)] = qdot_grad_together[:,:,k]
                        Gamma = torch.zeros(2,2,2)
                        Gamma[:,:,0] = Temp1[0].detach().to(self.device)
                        Gamma[:,:,1] = Temp1[1].detach().to(self.device)
                        Matrix_Gamma[2*k:2*(k+1),2*k:2*(k+1),2*k:2*(k+1)] = Gamma
                        Matrix_f[2*k:2*(k+1),:] = torch.reshape(current_output[:,k],[2,1])
                        Matrix_G[2*k:2*(k+1),2*k:2*(k+1)] = Temp2
                        Matrix_det_G[2*k:2*(k+1),2*k:2*(k+1)] = torch.sqrt(torch.det(Temp2))*torch.eye(2)
                        Matrix_G_inv[2*k:2*(k+1),2*k:2*(k+1)] = torch.inverse(Temp2)
                    ######################################
                    cov_der = Matrix_del_f+torch.tensordot(Matrix_Gamma,Matrix_f,dims=([2],[0])).reshape(2*num_sampled_batch,2*num_sampled_batch)
                    #print(cov_der)
                    Integral_approximation = torch.trace(torch.mm(torch.mm(torch.mm(torch.mm(cov_der.t(),Matrix_G),cov_der),Matrix_G_inv),Matrix_det_G))
                    total_loss_reg = Integral_approximation*self.dA/2*(self.nq1-1)/(self.n_totalgrid-1)*(self.nq2-1)/(self.n_totalgrid-1)
                    total_loss = (total_loss_reg+penalty*(self.penalty_fun(loss_task,eps,epoch,mode = penalty_mode))+penalty_boundary*((loss_boundary)**2)).data.clone().to(device_c)
                
                
                
                
                print ('\r epoch = '+str(t) +' i = '+str(i+1)+ 
                       ', loss = ' +str(ld.data.numpy())+
                       ', total_loss = ' +str(total_loss.data.numpy())+
                       ', loss_b = '+str(loss_boundary.to(device_c).data.numpy())+ 
                       ', loss_t = '+str(loss_task.to(device_c).data.numpy())+
                       ', loss_reg = '+str(loss_reg.to(device_c).data.numpy())+
                       ', loss_reg_total =  '+str(total_loss_reg.to(device_c).data.numpy())+'           .',end = ' ')
                if t>10000:
                    if (loss_mat[t]-loss_mat[t-100])/loss_mat[t] < 0.00001:
                        print('Optimization finished')
                
        
    def optim_Kinetic_reg(self, robot, q_in_reg, qtraj, q_dot,dt,q_in_boundary,q_dot_boundary,penalty,penalty_boundary, Xstable, learning_rate = 1e-3, epoch=10000,
                      batch_size = 400,penalty_mode = 0):
        
        q_in_small = self.get_qsmall(self.n_totalgrid,self.n_totalgrid,self.qmin,self.qmax)
        q_in_small = q_in_small.to(self.device)
        q_in_small.requires_grad = True
        
        
        dataset = Grid_dataset(q_in_reg.numpy())
        train_loader = DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)
        if batch_size>q_in_reg.shape[0]:
            #print('batch_size is reduced to maximum batch_size = '+str(q_in_reg.shape[0])+'.')
            batch_size = q_in_reg.shape[0]
            
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        #loss_mat = torch.zeros(epoch,dtype=dtype,device=self.device)

        for t in range(epoch):
            for i, data in enumerate(train_loader):
                
                qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
                temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
                temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
                temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
                
                loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
                penalty_boundary = 10000               #####################################

                data0 = data[0]
                data1 = data[1]
                current_grid_data = torch.zeros(data[0].shape[0],2, device=self.device, dtype=dtype)
                current_grid_data[:,0]  = data[0][:,0].clone()
                current_grid_data[:,1]  = data[1][:,0].clone()
                current_grid_data.requires_grad=True

                current_output = self.model.f_forward(current_grid_data,Xstable).to(self.device)
                current_output = current_output.t()
                co1 = torch.sum(current_output[:,0])
                co2 = torch.sum(current_output[:,1])
                qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
                qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]
                qdot_grad_together = torch.zeros(2,2,current_grid_data.shape[0]).to(self.device)
                qdot_grad_together[:,0,:] = qdot_grad1.t()
                qdot_grad_together[:,1,:] = qdot_grad2.t()

                ####################################### 
                num_sampled_batch = current_grid_data.shape[0]
                Matrix_del_f = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_Gamma = torch.zeros(2*num_sampled_batch,2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_f = torch.zeros(2*num_sampled_batch,1).to(self.device)
                Matrix_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_det_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                Matrix_G_inv = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)

                for k in range(num_sampled_batch):
                    Temp1 = robot.get_Christoffel_kinetic(current_grid_data[k,:])
                    Temp2 = robot.get_Kinetic_Riemannian_metric(current_grid_data[k,:])

                    Matrix_del_f[2*k:2*(k+1),2*k:2*(k+1)] = qdot_grad_together[:,:,k]
                    Gamma = torch.zeros(2,2,2)
                    Gamma[:,:,0] = Temp1[0].detach().to(self.device)
                    Gamma[:,:,1] = Temp1[1].detach().to(self.device)
                    Matrix_Gamma[2*k:2*(k+1),2*k:2*(k+1),2*k:2*(k+1)] = Gamma
                    Matrix_f[2*k:2*(k+1),:] = torch.reshape(current_output[:,k],[2,1])
                    Matrix_G[2*k:2*(k+1),2*k:2*(k+1)] = Temp2
                    Matrix_det_G[2*k:2*(k+1),2*k:2*(k+1)] = torch.sqrt(torch.det(Temp2))*torch.eye(2)
                    Matrix_G_inv[2*k:2*(k+1),2*k:2*(k+1)] = torch.inverse(Temp2)
                ######################################
                cov_der = Matrix_del_f+torch.tensordot(Matrix_Gamma,Matrix_f,dims=([2],[0])).reshape(2*num_sampled_batch,2*num_sampled_batch)
                Integral_approximation = torch.trace(torch.mm(torch.mm(torch.mm(torch.mm(cov_der.t(),Matrix_G),cov_der),Matrix_G_inv),Matrix_det_G))

                loss_reg = Integral_approximation*self.dA/2*(self.nq1*self.nq2/batch_size)


                qdot_pred = self.model.f_forward(q_in,Xstable).to(self.device)
                loss_task = dt*loss_fn(qdot_pred,qdot_real).to(self.device)
                penalty = 10000                             ##########################################################################
                eps = 0.1
                loss = loss_reg+penalty*(self.penalty_fun(loss_task,eps,epoch,mode = penalty_mode))+penalty_boundary*((loss_boundary)**2)
                #print(loss)
                loss.backward(retain_graph=True)
                
                max_grad_norm = 1
                norm_type=2
            
                if max_grad_norm > 0:
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
                        
                optimizer.step()
                q_in.grad.zero_()
                self.model.zero_grad()
                optimizer.zero_grad()
                current_grid_data.grad.zero_()
                ld = loss.data.clone().to(device_c)
                #loss_mat[t] = loss.data
                ##################################################################################
                if i%10==0:
                    total_grid = q_in_small
                    current_output = self.model.f_forward(total_grid,self.Xstable).to(self.device)
                    current_output = current_output.t()
                    co1 = torch.sum(current_output[:,0])
                    co2 = torch.sum(current_output[:,1])
                    qdot_grad1 = grad(co1,total_grid,create_graph=True,retain_graph=True)[0]
                    qdot_grad2 = grad(co2,total_grid,create_graph=True,retain_graph=True)[0]
                    qdot_grad_together = torch.zeros(2,2,total_grid.shape[0]).to(self.device)
                    qdot_grad_together[:,0,:] = qdot_grad1.t()
                    qdot_grad_together[:,1,:] = qdot_grad2.t()
                    
                    num_sampled_batch = total_grid.shape[0]
                    Matrix_del_f = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_Gamma = torch.zeros(2*num_sampled_batch,2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_f = torch.zeros(2*num_sampled_batch,1).to(self.device)
                    Matrix_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_det_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    Matrix_G_inv = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
                    
                    for k in range(num_sampled_batch):
                        Temp1 = robot.get_Christoffel_kinetic(total_grid[k,:])
                        Temp2 = robot.get_Kinetic_Riemannian_metric(total_grid[k,:])

                        Matrix_del_f[2*k:2*(k+1),2*k:2*(k+1)] = qdot_grad_together[:,:,k]
                        Gamma = torch.zeros(2,2,2)
                        Gamma[:,:,0] = Temp1[0].detach().to(self.device)
                        Gamma[:,:,1] = Temp1[1].detach().to(self.device)
                        Matrix_Gamma[2*k:2*(k+1),2*k:2*(k+1),2*k:2*(k+1)] = Gamma
                        Matrix_f[2*k:2*(k+1),:] = torch.reshape(current_output[:,k],[2,1])
                        Matrix_G[2*k:2*(k+1),2*k:2*(k+1)] = Temp2
                        Matrix_det_G[2*k:2*(k+1),2*k:2*(k+1)] = torch.sqrt(torch.det(Temp2))*torch.eye(2)
                        Matrix_G_inv[2*k:2*(k+1),2*k:2*(k+1)] = torch.inverse(Temp2)
                    ######################################
                    cov_der = Matrix_del_f+torch.tensordot(Matrix_Gamma,Matrix_f,dims=([2],[0])).reshape(2*num_sampled_batch,2*num_sampled_batch)
                    #print(cov_der)
                    Integral_approximation = torch.trace(torch.mm(torch.mm(torch.mm(torch.mm(cov_der.t(),Matrix_G),cov_der),Matrix_G_inv),Matrix_det_G))
                    total_loss_reg = Integral_approximation*self.dA/2*(self.nq1-1)/(self.n_totalgrid-1)*(self.nq2-1)/(self.n_totalgrid-1)
                    total_loss = (total_loss_reg+penalty*(self.penalty_fun(loss_task,eps,epoch,mode = penalty_mode))+penalty_boundary*((loss_boundary)**2)).data.clone().to(device_c)
                    
                    
                    
                    
                    
                    
                    
                    #total_grid = q_in_reg.to(self.device)
                    #total_grid.requires_grad=True
                    #current_output = self.model.f_forward(total_grid,self.Xstable).to(self.device)
                    
                    #co1 = torch.sum(current_output[:,0])
                    #co2 = torch.sum(current_output[:,1])
                    
                    #qdot_grad1 = grad(co1,total_grid,create_graph=True,retain_graph=True)[0]
                    #qdot_grad2 = grad(co2,total_grid,create_graph=True,retain_graph=True)[0]
                    
                    #qdot_grad_norm = qdot_grad1**2+qdot_grad2**2
                    #total_loss_reg = torch.sum(qdot_grad_norm)*self.dA/2
                    #total_loss = (total_loss_reg+penalty*(F.relu(loss_task-eps)**2)+penalty_boundary*((loss_boundary)**2)).data.clone().to(device_c)
                    
                print ('\r epoch = '+str(t) +' i = '+str(i+1)+ 
                       ', loss = ' +str(ld.data.numpy())+
                       ', total_loss = ' +str(total_loss.data.numpy())+
                       ', loss_b = '+str(loss_boundary.to(device_c).data.numpy())+ 
                       ', loss_t = '+str(loss_task.to(device_c).data.numpy())+
                       ', loss_reg = '+str(loss_reg.to(device_c).data.numpy())+
                       ', loss_reg_total =  '+str(total_loss_reg.to(device_c).data.numpy())+'           .',end = ' ')
                if t>10000:
                    if (loss_mat[t]-loss_mat[t-100])/loss_mat[t] < 0.00001:
                        print('Optimization finished')
            
        
    def performance_metric(self,robot, q_in_reg, qtraj, q_dot,dt,q_in_boundary,q_dot_boundary, Xstable,alpha_kinematic):
        #1 trajectory ######################################################################
        q_in = qtraj.t().to(self.device)
        q_in.requires_grad=True
        qdot_real = q_dot.t().to(self.device)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        qdot_pred = self.model.f_forward(q_in,self.Xstable).to(self.device)
        loss_traj = dt*loss_fn(qdot_pred,qdot_real).data.to(self.device)
        loss_traj = loss_traj*(self.T**2)
        print("#1 done")
        #2 boundary######################################################################
        qdot_bound_pred = self.model.f_forward(q_in_boundary,self.Xstable).to(self.device)
        temp0_boundary = torch.sqrt((qdot_bound_pred**2)[:,0]+(qdot_bound_pred**2)[:,1])
        temp_boundary = torch.mul(q_dot_boundary,qdot_bound_pred)
        temp2_boundary = torch.sum(F.relu(-temp_boundary[:,0]-temp_boundary[:,1])/temp0_boundary)# -는 relu 쓰기위함
        loss_boundary = math.pi*4/(q_in_boundary.shape[0])*temp2_boundary 
        
        loss_boundary = loss_boundary*(self.T**2)
        print("#2 done")
        #3 Euclidean(alpha = 1)########################################################
        total_grid = q_in_reg.to(self.device)
        total_grid.requires_grad=True
        current_output = self.model.f_forward(total_grid,self.Xstable).to(self.device)
        
        co1 = torch.sum(current_output[:,0])
        co2 = torch.sum(current_output[:,1])
        
        qdot_grad1 = grad(co1,total_grid,retain_graph=True)[0]
        qdot_grad2 = grad(co2,total_grid,retain_graph=True)[0]
        
        qdot_grad_norm = qdot_grad1**2+qdot_grad2**2
        loss_Euc = torch.sum(qdot_grad_norm)*self.dA/2
        loss_Euc = loss_Euc.data.to(device_c)
        loss_Euc = loss_Euc*(self.T**2)
        print("#3 done")
        #4 Kinematic(alpha = 0)#########################################################
        loss_kinematic = 0
        loss_kinetic = 0
        batch_size = 40
        epoch = 1
        dataset = Grid_dataset(q_in_reg.numpy())
        train_loader = DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=8)
        for i, data in enumerate(train_loader):
            data0 = data[0]
            data1 = data[1]
            current_grid_data = torch.zeros(data[0].shape[0],2, device=self.device, dtype=dtype)
            current_grid_data[:,0]  = data[0][:,0].clone()
            current_grid_data[:,1]  = data[1][:,0].clone()
            current_grid_data.requires_grad=True

            current_output = self.model.f_forward(current_grid_data,Xstable).to(self.device)
            current_output = current_output.t()
            co1 = torch.sum(current_output[:,0])
            co2 = torch.sum(current_output[:,1])
            qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
            qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]
            qdot_grad_together = torch.zeros(2,2,current_grid_data.shape[0]).to(self.device)
            qdot_grad_together[:,0,:] = qdot_grad1.t()
            qdot_grad_together[:,1,:] = qdot_grad2.t()
            
            num_sampled_batch = current_grid_data.shape[0]
            Matrix_del_f = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_Gamma = torch.zeros(2*num_sampled_batch,2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_f = torch.zeros(2*num_sampled_batch,1).to(self.device)
            Matrix_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_det_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_G_inv = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            for k in range(num_sampled_batch):
                Temp1 = robot.get_Christoffel_kinematic(current_grid_data[k,:],alpha_kinematic)
                Temp2 = robot.get_Kinematic_Riemannian_metric(current_grid_data[k,:],alpha_kinematic)

                Matrix_del_f[2*k:2*(k+1),2*k:2*(k+1)] = qdot_grad_together[:,:,k]
                Gamma = torch.zeros(2,2,2)
                Gamma[:,:,0] = Temp1[0].detach().to(self.device)
                Gamma[:,:,1] = Temp1[1].detach().to(self.device)
                Matrix_Gamma[2*k:2*(k+1),2*k:2*(k+1),2*k:2*(k+1)] = Gamma
                Matrix_f[2*k:2*(k+1),:] = torch.reshape(current_output[:,k],[2,1])
                Matrix_G[2*k:2*(k+1),2*k:2*(k+1)] = Temp2
                Matrix_det_G[2*k:2*(k+1),2*k:2*(k+1)] = torch.sqrt(torch.det(Temp2))*torch.eye(2)
                Matrix_G_inv[2*k:2*(k+1),2*k:2*(k+1)] = torch.inverse(Temp2)
            cov_der = Matrix_del_f+torch.tensordot(Matrix_Gamma,Matrix_f,dims=([2],[0])).reshape(2*num_sampled_batch,2*num_sampled_batch)
            Integral_approximation = torch.trace(torch.mm(torch.mm(torch.mm(torch.mm(cov_der.t(),Matrix_G),cov_der),Matrix_G_inv),Matrix_det_G))
            loss_kinematic_current = Integral_approximation*self.dA/2
            loss_kinematic += loss_kinematic_current.data
            print(Integral_approximation)
        
        loss_kinematic = loss_kinematic*(self.T**2)
        print("#4 done")

                
        #5 Kinetic #########################################################
        loss_kinetic = 0
        for i, data in enumerate(train_loader):
            data0 = data[0]
            data1 = data[1]
            current_grid_data = torch.zeros(data[0].shape[0],2, device=self.device, dtype=dtype)
            current_grid_data[:,0]  = data[0][:,0].clone()
            current_grid_data[:,1]  = data[1][:,0].clone()
            current_grid_data.requires_grad=True
            current_output = self.model.f_forward(current_grid_data,Xstable).to(self.device)
            current_output = current_output.t()
            co1 = torch.sum(current_output[:,0])
            co2 = torch.sum(current_output[:,1])
            qdot_grad1 = grad(co1,current_grid_data,create_graph=True,retain_graph=True)[0]
            qdot_grad2 = grad(co2,current_grid_data,create_graph=True,retain_graph=True)[0]
            qdot_grad_together = torch.zeros(2,2,current_grid_data.shape[0]).to(self.device)
            qdot_grad_together[:,0,:] = qdot_grad1.t()
            qdot_grad_together[:,1,:] = qdot_grad2.t()

            ####################################### 
            num_sampled_batch = current_grid_data.shape[0]
            Matrix_del_f = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_Gamma = torch.zeros(2*num_sampled_batch,2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_f = torch.zeros(2*num_sampled_batch,1).to(self.device)
            Matrix_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_det_G = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)
            Matrix_G_inv = torch.zeros(2*num_sampled_batch,2*num_sampled_batch).to(self.device)

            for k in range(num_sampled_batch):
                Temp1 = robot.get_Christoffel_kinetic(current_grid_data[k,:])
                Temp2 = robot.get_Kinetic_Riemannian_metric(current_grid_data[k,:])

                Matrix_del_f[2*k:2*(k+1),2*k:2*(k+1)] = qdot_grad_together[:,:,k]
                Gamma = torch.zeros(2,2,2)
                Gamma[:,:,0] = Temp1[0].detach().to(self.device)
                Gamma[:,:,1] = Temp1[1].detach().to(self.device)
                Matrix_Gamma[2*k:2*(k+1),2*k:2*(k+1),2*k:2*(k+1)] = Gamma
                Matrix_f[2*k:2*(k+1),:] = torch.reshape(current_output[:,k],[2,1])
                Matrix_G[2*k:2*(k+1),2*k:2*(k+1)] = Temp2
                Matrix_det_G[2*k:2*(k+1),2*k:2*(k+1)] = torch.sqrt(torch.det(Temp2))*torch.eye(2)
                Matrix_G_inv[2*k:2*(k+1),2*k:2*(k+1)] = torch.inverse(Temp2)
            ######################################
            cov_der = Matrix_del_f+torch.tensordot(Matrix_Gamma,Matrix_f,dims=([2],[0])).reshape(2*num_sampled_batch,2*num_sampled_batch)
            Integral_approximation = torch.trace(torch.mm(torch.mm(torch.mm(torch.mm(cov_der.t(),Matrix_G),cov_der),Matrix_G_inv),Matrix_det_G))

            loss_kinetic_current = Integral_approximation*self.dA/2
            loss_kinetic += loss_kinetic_current.data
        print("#5 done")
        
        loss_kinetic = loss_kinetic*(self.T**2)
        return loss_traj, loss_boundary, loss_Euc, loss_kinematic, loss_kinetic

        
        
    
    
    
        
        