import torch
import torch.nn as nn
import numpy as np
import random                                                                  
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.deterministic=True
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
import math
dtype = torch.float
device_c = torch.device("cpu")
device = torch.device("cuda:0")

class ICNN_plot():
    def __init__(self, model,qmin,qmax,nq1,nq2,xmin,xmax,nx1,nx2,Xstable,qtraj,l1,l2):
        #이병호 추가: 플롯 가로세로 한것들
        self.fswitch = 1
        self.nq1 = nq1
        self.nq2 = nq2
        ## added by YH LEE
        self.xmin = xmin
        self.xmax = xmax
        self.nx1 = nx1
        self.nx2 = nx2
        ##
        self.qtraj = qtraj
        self.q1_0 = (qmax[0]-qmin[0])/(2*nq1)+qmin[0]
        self.q2_0 = (qmax[1]-qmin[1])/(2*nq2)+qmin[1]
        self.q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1)
        self.q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2)
        self.q1 = torch.linspace(qmin[0],qmax[0],nq1)#(self.q1_0,self.q1_f,nq1)
        self.q2 = torch.linspace(qmin[1],qmax[1],nq1)#(self.q2_0,self.q2_f,nq2)
        self.q1_mesh,self.q2_mesh = torch.meshgrid(self.q1,self.q2)
        
        self.x1_0 = (xmax[0]-xmin[0])/(2*nx1)+xmin[0]
        self.x2_0 = (xmax[1]-xmin[1])/(2*nx2)+xmin[1]
        self.x1_f = xmax[0] - (xmax[0]-xmin[0])/(2*nx1)
        self.x2_f = xmax[1] - (xmax[1]-xmin[1])/(2*nx2)
        self.x1 = torch.linspace(self.x1_0,self.x1_f,nx1)
        self.x2 = torch.linspace(self.x2_0,self.x2_f,nx2)
        self.x1_mesh,self.x2_mesh = torch.meshgrid(self.x1,self.x2)
        self.xmesh = torch.zeros((nx1,nx2,2), device=device, dtype=dtype)
        self.xmesh[:,:,0] = self.x1_mesh
        self.xmesh[:,:,1] = self.x2_mesh
        
        
        self.Xstable = Xstable
        self.f_total = torch.zeros((nq2, nq1,2), device=torch.device("cpu"), dtype=dtype)
        self.f_grad_total = torch.zeros((nq2, nq1), device=torch.device("cpu"), dtype=dtype)
        self.fh_total = torch.zeros((nq2, nq1,2), device=torch.device("cpu"), dtype=dtype)
        self.V_total = torch.zeros((nq2, nq1), device=torch.device("cpu"), dtype=dtype)
        #self.Vgrad_total = torch.zeros((nq2, nq1,2), device=torch.device("cpu"), dtype=dtype)
        self.model = model
        
        self.qmesh = self.get_mesh(nq1,nq2,qmin,qmax)
        self.l1 = l1
        self.l2 = l2
        
    def get_mesh(self,nq1,nq2,qmin,qmax):  #중복되니까 삭제하자
        q_in_mesh = torch.zeros((nq1,nq2,2), device=device, dtype=dtype)
        q1_0 = (qmax[0]-qmin[0])/(2*nq1)+qmin[0]
        q2_0 = (qmax[1]-qmin[1])/(2*nq2)+qmin[1]
        q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1)
        q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2)
        q1 = torch.linspace(qmin[0],qmax[0],nq1)#(q1_0,q1_f,nq1)
        q2 = torch.linspace(qmin[1],qmax[1],nq2)#(q2_0,q2_f,nq2)
        q1_mesh,q2_mesh = torch.meshgrid(q1,q2)
        q_in_mesh[:,:,0] = q1_mesh
        q_in_mesh[:,:,1] = q2_mesh
        #q_in_reg = q_in_mesh.view(-1,2).to(device_c)
        #qmesh2 = q_in_reg.view(nq1,nq2,2)
        #print(torch.sum(q_in_mesh-qmesh2.to(device)))
        return q_in_mesh
    
    
    def cal_f(self):
        Xlong = self.qmesh.view(-1,2).to(device)
        Xlong.requires_grad=True
        ftotal_long = self.model.f_forward(Xlong, self.Xstable)
        self.f_total = ftotal_long.view(self.nq1,self.nq2,2).to(device_c).detach()
        co1 = torch.sum(ftotal_long[:,0])
        co2 = torch.sum(ftotal_long[:,1])
        qdot_grad1 = grad(co1,Xlong,create_graph=True,retain_graph=True)[0]
        qdot_grad2 = grad(co2,Xlong,create_graph=True,retain_graph=True)[0]
        qdot_grad_norm1 = torch.sqrt((qdot_grad1**2)[:,0]+(qdot_grad1**2)[:,1]+(qdot_grad2**2)[:,0]+(qdot_grad2**2)[:,1])
        self.f_grad_total = qdot_grad_norm1.view(self.nq1,self.nq2).to(device_c).detach()
        #print(qdot_grad_norm1.view(self.nq1,self.nq2))
        
        #for i in range(self.nq1):
        #    #X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=device, dtype=dtype, requires_grad=True)
        #    
        #    for j in range(self.nq2):
        #        X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=device, dtype=dtype, requires_grad=True)
        #        f_current = self.model.f_forward(X, self.Xstable)
        #        self.f_total[i,j,:] = f_current
        #        
        #        co1 = f_current[0,0]
        #        co2 = f_current[0,1]
        #        qdot_grad1 = grad(co1,X,create_graph=True,retain_graph=True)[0]
        #        qdot_grad2 = grad(co2,X,create_graph=True,retain_graph=True)[0]
        #        qdot_grad_norm = torch.sqrt((qdot_grad1**2)[:,0]+(qdot_grad1**2)[:,1]+(qdot_grad2**2)[:,0]+(qdot_grad2**2)[:,1])
        #        color_current = torch.sum(qdot_grad_norm)
        #        self.f_grad_total[i,j] = color_current
        #print(self.f_grad_total)
        #print(self.ftotal - self.f_total)
               
                    
                    
    def inverse_kinematics_vec(self,x):
        x1 = x[:,0]
        x2 = x[:,1]
        q2 = np.arccos((x1**2+x2**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2))
        #print(q2)
        q1 = np.arccos(((self.l1+self.l2*np.cos(q2))*x1+(self.l2*np.sin(q2))*x2)/(self.l1**2+self.l2**2+2*self.l1*self.l2*np.cos(q2)))
        #print(q1)
        q = torch.zeros(x.shape[0],2,dtype=dtype)#np.zeros([x.shape[0],2])
        q[:,0] = q1
        q[:,1] = q2
        return q              
    
    def get_jacobian_vec(self, q):
        Jacobian = torch.zeros(q.shape[0],2,2)
        Jacobian[:,0,0] = -self.l1*torch.sin(q[:,0])-self.l2*torch.sin(q[:,0]+q[:,1])
        Jacobian[:,1,0] = self.l1*torch.cos(q[:,0])+self.l2*torch.cos(q[:,0]+q[:,1])
        Jacobian[:,0,1] = -self.l2*torch.sin(q[:,0]+q[:,1])
        Jacobian[:,1,1] = self.l2*torch.cos(q[:,0]+q[:,1])
        return Jacobian 

        
        
    
    def cal_fhat(self):
        for i in range(self.nq1):
            for j in range(self.nq2):
                X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=device, dtype=dtype, requires_grad=True)
                fh_current = self.model.fhat_forward(X)
                self.fh_total[i,j,:] = fh_current
                
                
    def cal_V(self):
        for i in range(self.nq1):
            for j in range(self.nq2):
                X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=device, dtype=dtype, requires_grad=True)
                V_current = self.model.V_forward(X, self.Xstable)
                a = V_current.clone()
                self.V_total[i,j] = a
                
    def plot_f(self,filename, density, widthscale, widthbase, quiver = False, streamplot = True):

        self.cal_f()

        pi = math.pi
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        fnumpy = self.f_total.detach()
        U = fnumpy[:,:,0].t().numpy()
        V = fnumpy[:,:,1].t().numpy()
        
        #f streamline
        if(quiver):
            plt.quiver(self.q1_mesh,self.q2_mesh,self.f_total[:,:,0].detach(),self.f_total[:,:,1].detach(),color = 'r',width=0.002)
        if(streamplot):
            #print(np.sqrt(U*U+V*V))
            #print(np.ones([self.nq1,self.nq2]))
            width = np.sqrt(U*U+V*V)*widthscale+widthbase*np.ones(np.shape(U))
            #width[width>10] = 10
            #print(width)
            color = self.f_grad_total.detach().t().numpy()
            #print(color)
            strm = plt.streamplot(self.q1_mesh.t().numpy(),self.q2_mesh.t().numpy(),fnumpy[:,:,0].t().numpy(),fnumpy[:,:,1].t().numpy(),
                           density = density,color = color,linewidth = width, cmap='autumn')
            plt.colorbar(strm.lines)

        plt.savefig(filename,dpi = 500)
    
    
    def cal_ftask(self):
        xlong = self.xmesh.view(-1,2).to(device_c)
        x1 = xlong[:,0]
        x2 = xlong[:,1]
        U = np.zeros([self.nx1,self.nx2])
        V = np.zeros([self.nx1,self.nx2])
        
        #xlong_valid = 
        #qlong = self.inverse_kinematics_vec(xlong)
        a = (self.l1-self.l2)**2 < x1**2 + x2**2
        b = x1**2 + x2**2 < (self.l1+self.l2)**2
        c = self.l2**2 < (x1-self.l1)**2+x2**2
        d = (x1+self.l1)**2 + x2**2 < self.l2**2
        e = x2 > 0
        f = d+e
        f[f>1] = 1 #No meaning
        g = a*b*c*f
        qlong = self.Xstable.to(device_c)*torch.ones(xlong.shape)
        xlong_valid = xlong[g>0]
        h = 0.0001
        qvalid = self.inverse_kinematics_vec(xlong_valid).to(device)
        qvalid.requires_grad=True
        qvalidx = qvalid+torch.tensor([h,0],device=device)
        qvalidx_ = qvalid+torch.tensor([-h,0],device=device)
        qvalidy = qvalid+torch.tensor([0,h],device=device)
        qvalidy_ = qvalid+torch.tensor([0,-h],device=device)
        #print(qvalid-qvalidx)
        
        
        Vec = self.model.f_forward(qvalid,self.Xstable).cpu().detach()
        Vecx = self.model.f_forward(qvalidx,self.Xstable).cpu().detach()
        Vecx_= self.model.f_forward(qvalidx_,self.Xstable).cpu().detach()
        Vecy = self.model.f_forward(qvalidy,self.Xstable).cpu().detach()
        Vecy_= self.model.f_forward(qvalidy_,self.Xstable).cpu().detach()
        
        
        J = self.get_jacobian_vec(qvalid).detach()
        Jx = self.get_jacobian_vec(qvalidx).detach()
        Jx_ = self.get_jacobian_vec(qvalidx_).detach()
        Jy = self.get_jacobian_vec(qvalidy).detach()
        Jy_ = self.get_jacobian_vec(qvalidy_).detach()
        
    
        f_task_valid = torch.zeros(qvalid.shape)
        f_task = torch.zeros(xlong.shape)
        self.ftask = f_task.view(self.nx1,self.nx2,2).detach().numpy()
        temp1 = torch.zeros(qvalid.shape,dtype=dtype)
        temp2 = torch.zeros(qvalid.shape,dtype=dtype)
        #print(self.ftask)
        #print(J)
        for i in range(qvalid.shape[0]):
            x_dot = torch.matmul(J[i],Vec[i].view(2,1))[:,0]
            x_dotx = torch.matmul(Jx[i],Vecx[i].view(2,1))[:,0]
            x_dotx_ = torch.matmul(Jx_[i],Vecx_[i].view(2,1))[:,0]
            x_doty = torch.matmul(Jy[i],Vecy[i].view(2,1))[:,0]
            x_doty_ = torch.matmul(Jy_[i],Vecy_[i].view(2,1))[:,0]
            temp1[i] = (x_dotx-x_dotx_)/(2*h)
            temp2[i] = (x_doty-x_doty_)/(2*h)
            
            
            f_task_valid[i] = x_dot
            #print(x_dot)
        #x_dot = np.matmul(J,np.reshape(Vec_num,[2,1]))
        #print(temp1,temp2)
        self.f_task_grad_valid = torch.sqrt((temp1**2)[:,0]+(temp1**2)[:,1]+(temp2**2)[:,0]+(temp2**2)[:,1])
        self.f_task_grad = torch.zeros(qlong.shape[0],dtype=dtype)
        self.f_task_grad[g>0] = self.f_task_grad_valid
        self.f_task_grad= self.f_task_grad.view(self.nx1,self.nx2)
        f_task[g>0] = f_task_valid
        #print(self.f_task_grad)
        #print(self.ftask)
        #print(f_task)
        
        
    def plot_fhat(self,filename, density, quiver = False, streamplot = True):
        pi = math.pi
        self.cal_fhat()
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        
        fhnumpy = self.fh_total.detach()
        U2 = fhnumpy[:,:,0].t().numpy()
        V2 = fhnumpy[:,:,1].t().numpy()
        
        if(quiver):
            plt.quiver(self.q1_mesh,self.q2_mesh,self.fh_total[:,:,0].detach(),self.fh_total[:,:,1].detach(),color = 'b',width=0.0015)
        if(streamplot):
            plt.streamplot(self.q1_mesh.t().numpy(),self.q2_mesh.t().numpy(),fhnumpy[:,:,0].t().numpy(),fhnumpy[:,:,1].t().numpy(),
                           density = density,color = np.sqrt(U2*U2+V2*V2))
        
        plt.savefig(filename,dpi = 500)
        
    def plot_V(self, filename,num_level):
        pi = math.pi
        self.cal_V()
            
        
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        
        Z = self.V_total.detach()
        plt.contour(self.q1_mesh,self.q2_mesh,Z,
                   levels = np.linspace(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max(), num_level))

        plt.savefig(filename,dpi = 500)


        
    def plot_f_taskspace(self,filename, robot, density, widthscale, widthbase, quiver = False, streamplot = True):

        l1 = robot.l1
        l2 = robot.l2
        theta = np.linspace(0,math.pi,num = 100)
        x_r1 = (l1-l2)*np.cos(theta)
        y_r1 = (l1-l2)*np.sin(theta)
        x_r2 = (l1+l2)*np.cos(theta)
        y_r2 = (l1+l2)*np.sin(theta)
        if (l1-l2)>0:
            plt.plot(x_r1,y_r1,'--b')
        plt.plot(x_r2,y_r2,'--b')
        x_r3 = -(l1)+l2*np.cos(theta+math.pi)
        y_r3 = l2*np.sin(theta+math.pi)
        x_r4 = (l1)+l2*np.cos(theta)
        y_r4 = l2*np.sin(theta)
        plt.plot(x_r3,y_r3,'--b')
        plt.plot(x_r4,y_r4,'--b')
        #이병호 추가: 플롯 가로세로
        axis_lengths=  [(self.xmax[0]-self.xmin[0]),(self.xmax[1]-self.xmin[1])]
        max_length  = max(axis_lengths)
        graphs_axis_length2 = 10*axis_lengths[1]/max_length
        graphs_axis_length1 = 10*axis_lengths[0]/max_length + 0.23*graphs_axis_length2
        #print(graphs_axis_length1,graphs_axis_length2)
        plt.rcParams["figure.figsize"] = (graphs_axis_length1,graphs_axis_length2)
        plt.plot((robot.xinit.numpy()[0],robot.xfinal.numpy()[0]),(robot.xinit.numpy()[1],robot.xfinal.numpy()[1]),'g')
        plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])
        
        self.cal_ftask()
        
        U = self.ftask[:,:,0]
        V = self.ftask[:,:,1]
        #print(U,V)
        
        
        
        #for i in range(self.x1_mesh.shape[0]):
        #    for j in range(self.x1_mesh.shape[1]):
        #        x1 = self.x1_mesh[i,j]
        #        x2 = self.x2_mesh[i,j]       
        #        a = (robot.l1-robot.l2)**2 < x1**2 + x2**2
        #        b = x1**2 + x2**2 < (robot.l1+robot.l2)**2
        #        c = robot.l2**2 < (x1-robot.l1)**2+x2**2
        #        d = (x1+robot.l1)**2 + x2**2 < robot.l2**2
        #        e = x2 > 0
#
        #        if a.data.numpy()*b.data.numpy()*c.data.numpy()*(d.data.numpy()+e.data.numpy())>0:
        #            Temp = robot.inverse_kinematics([x1,x2])
        #            Vec = self.model.f_forward(torch.tensor([np.reshape(Temp,[2,])], device=device, dtype=dtype, requires_grad=True),self.Xstable)
        #            Vec_num = Vec.data.cpu().numpy()
        #            J = robot.get_jacobian(torch.tensor(np.reshape(Temp,[2,])))
        #            x_dot = np.matmul(J,np.reshape(Vec_num,[2,1]))
        #            U[i,j] = x_dot[0,:]
        #            V[i,j] = x_dot[1,:]
                    

        if(quiver):
            plt.quiver(self.x1_mesh.numpy(), self.x2_mesh.numpy(), U, V,'r',width=0.003)
        if(streamplot):
            
            width = np.sqrt(U*U+V*V)*widthscale+widthbase*np.ones(np.shape(U))
            #width[width>10] = 10
            #print(width)
            color = self.f_task_grad.detach().t().numpy()
            
            strm = plt.streamplot(self.x1_mesh.t().numpy(), self.x2_mesh.t().numpy(), np.transpose(U), np.transpose(V), density = density ,color =color,linewidth = width,
                                  cmap='autumn')
            plt.colorbar(strm.lines)
        plt.show()
        plt.savefig(filename,dpi = 500)
       