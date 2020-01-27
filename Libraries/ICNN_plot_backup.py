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
    def __init__(self, model,qmin,qmax,nq1,nq2,xmin,xmax,nx1,nx2,Xstable,qtraj):
        self.nq1 = nq1
        self.nq2 = nq2
        ## added by YH LEE
        self.xmin = xmin
        self.xmax = xmax
        self.nx1 = nx1
        self.nx2 = nx2
        
        
        self.x1_0 = (xmax[0]-xmin[0])/(2*nx1)+xmin[0]
        self.x2_0 = (xmax[1]-xmin[1])/(2*nx2)+xmin[1]
        self.x1_f = xmax[0] - (xmax[0]-xmin[0])/(2*nx1)
        self.x2_f = xmax[1] - (xmax[1]-xmin[1])/(2*nx2)
        self.x1 = torch.linspace(self.x1_0,self.x1_f,nx1)
        self.x2 = torch.linspace(self.x2_0,self.x2_f,nx2)
        self.x1_mesh,self.x2_mesh = torch.meshgrid(self.x1,self.x2)
        
        self.qtraj = qtraj
        self.q1_0 = (qmax[0]-qmin[0])/(2*nq1)+qmin[0]
        self.q2_0 = (qmax[1]-qmin[1])/(2*nq2)+qmin[1]
        self.q1_f = qmax[0] - (qmax[0]-qmin[0])/(2*nq1)
        self.q2_f = qmax[1] - (qmax[1]-qmin[1])/(2*nq2)
        self.q1 = torch.linspace(self.q1_0,self.q1_f,nq1)
        self.q2 = torch.linspace(self.q2_0,self.q2_f,nq2)
        self.q1_mesh,self.q2_mesh = torch.meshgrid(self.q1,self.q2)
        
        
        
        
        self.Xstable = Xstable
        self.f_total = torch.zeros((nq2, nq1,2), device=torch.device("cpu"), dtype=dtype)
        self.fh_total = torch.zeros((nq2, nq1,2), device=torch.device("cpu"), dtype=dtype)
        self.V_total = torch.zeros((nq2, nq1), device=torch.device("cpu"), dtype=dtype)
        #self.Vgrad_total = torch.zeros((nq2, nq1,2), device=torch.device("cpu"), dtype=dtype)
        self.model = model
    def cal_f(self):
        for i in range(self.nq1):
            for j in range(self.nq2):
                X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=device, dtype=dtype, requires_grad=True)
                f_current = self.model.f_forward(X, self.Xstable)
                self.f_total[i,j,:] = f_current
                
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
                
    def plot_f(self,filename, quiver = False, streamplot = True):
        pi = math.pi
        self.cal_f()
        plt.rcParams["figure.figsize"] = (15+4.5,15)
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
            plt.streamplot(self.q1_mesh.t().numpy(),self.q2_mesh.t().numpy(),fnumpy[:,:,0].t().numpy(),fnumpy[:,:,1].t().numpy(),
                           density = 3,color = np.log(U*U+V*V))

        plt.savefig(filename,dpi = 500)
    
    def plot_fhat(self,filename, quiver = False, streamplot = True):
        pi = math.pi
        self.cal_fhat()
        plt.rcParams["figure.figsize"] = (15,15)
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
                           density = 3,color = np.log(U2*U2+V2*V2))
        
        plt.savefig(filename,dpi = 500)
        
    def plot_V(self, filename):
        pi = math.pi
        self.cal_V()
        plt.rcParams["figure.figsize"] = (15,15)
        plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        
        Z = self.V_total.detach()
        plt.contour(self.q1_mesh,self.q2_mesh,Z,
                   levels = np.linspace(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max(), 50))

        plt.savefig(filename,dpi = 500)
        
        
    def plot_f_taskspace(self,filename, robot, density, quiver = False, streamplot = True):
        self.cal_f()
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

        plt.rcParams["figure.figsize"] = (12,7)
        plt.plot((robot.xinit.numpy()[0],robot.xfinal.numpy()[0]),(robot.xinit.numpy()[1],robot.xfinal.numpy()[1]),'g')
        plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])

        U = np.zeros([self.nx1,self.nx2])
        V = np.zeros([self.nx1,self.nx2])

        for i in range(self.x1_mesh.shape[0]):
            for j in range(self.x1_mesh.shape[1]):
                x1 = self.x1_mesh[i,j]
                x2 = self.x2_mesh[i,j]       
                a = (robot.l1-robot.l2)**2 < x1**2 + x2**2
                b = x1**2 + x2**2 < (robot.l1+robot.l2)**2
                c = robot.l2**2 < (x1-robot.l1)**2+x2**2
                d = (x1+robot.l1)**2 + x2**2 < robot.l2**2
                e = x2 > 0

                if a.data.numpy()*b.data.numpy()*c.data.numpy()*(d.data.numpy()+e.data.numpy())>0:
                    Temp = robot.inverse_kinematics([x1,x2])
                    Vec = self.model.f_forward(torch.tensor([np.reshape(Temp,[2,])], device=device, dtype=dtype, requires_grad=True),self.Xstable)
                    Vec_num = Vec.data.cpu().numpy()
                    J = robot.get_jacobian(torch.tensor(np.reshape(Temp,[2,])))
                    x_dot = np.matmul(J,np.reshape(Vec_num,[2,1]))
                    U[i,j] = x_dot[0,:]
                    V[i,j] = x_dot[1,:]
        if(quiver):
            plt.quiver(self.x1_mesh.numpy(), self.x2_mesh.numpy(), U, V,'r',width=0.003)
        if(streamplot):
            strm = plt.streamplot(self.x1_mesh.t().numpy(), self.x2_mesh.t().numpy(), np.transpose(U), np.transpose(V), density = density ,color = np.sqrt(U*U+V*V),
                     cmap='autumn')
            plt.colorbar(strm.lines)
         
        plt.savefig(filename,dpi = 500)
    def plot_only_f_taskspace(self,filename, robot, density, quiver = False, streamplot = True):
        self.cal_f()
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

        plt.rcParams["figure.figsize"] = (12,7)
        plt.plot((robot.xinit.numpy()[0],robot.xfinal.numpy()[0]),(robot.xinit.numpy()[1],robot.xfinal.numpy()[1]),'g')
        plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])

        #U = np.zeros([self.nx1,self.nx2])
        #V = np.zeros([self.nx1,self.nx2])

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
            strm = plt.streamplot(self.x1_mesh.t().numpy(), self.x2_mesh.t().numpy(), np.transpose(U), np.transpose(V), density = density ,color = np.sqrt(U*U+V*V),
                     cmap='autumn')
            plt.colorbar(strm.lines)
         
        plt.savefig(filename,dpi = 500)
