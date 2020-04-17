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
    def __init__(self, model,qmin,qmax,nq1,nq2,xmin,xmax,nx1,nx2,Xstable,qtraj,xtraj,l1,l2,device):
        #이병호 추가: 플롯 가로세로 한것들
        self.device = device
        self.fswitch = 1
        self.nq1 = nq1
        self.nq2 = nq2
        ## added by YH LEE
        self.xmin = xmin
        self.xmax = xmax
        self.nx1 = nx1
        self.nx2 = nx2
        self.qmin = qmin
        self.qmax = qmax
        ##
        self.xtraj = xtraj
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
        self.xmesh = torch.zeros((nx1,nx2,2), device=self.device, dtype=dtype)
        self.xmesh[:,:,0] = self.x1_mesh
        self.xmesh[:,:,1] = self.x2_mesh
        
        
        self.Xstable = Xstable
        #self.f_total = torch.zeros((nq2, nq1,2), device=device_c, dtype=dtype)
        #self.f_grad_total = torch.zeros((nq2, nq1), device=device_c, dtype=dtype)
        self.fh_total = torch.zeros((nq2, nq1,2), device=device_c, dtype=dtype)
        #self.V_total = torch.zeros((nq2, nq1), device=device_c, dtype=dtype)
        #self.Vgrad_total = torch.zeros((nq2, nq1,2), device=device_c, dtype=dtype)
        self.model = model
        
        self.qmesh = self.get_mesh(nq1,nq2,qmin,qmax)
        self.l1 = l1
        self.l2 = l2
        
    def get_mesh(self,nq1,nq2,qmin,qmax):  #중복되니까 삭제하자
        q_in_mesh = torch.zeros((nq1,nq2,2), device=self.device, dtype=dtype)
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
        #print(torch.sum(q_in_mesh-qmesh2.to(self.device)))
        return q_in_mesh
    
    
    def cal_f(self):
        Xlong = self.qmesh.view(-1,2).to(self.device)
        Xlong.requires_grad=True
        ftotal_long = self.model.f_forward(Xlong, self.Xstable)
        f_total = ftotal_long.view(self.nq1,self.nq2,2).to(device_c).detach()
        co1 = torch.sum(ftotal_long[:,0])
        co2 = torch.sum(ftotal_long[:,1])
        qdot_grad1 = grad(co1,Xlong,create_graph=True,retain_graph=True)[0]
        qdot_grad2 = grad(co2,Xlong,create_graph=True,retain_graph=True)[0]
        qdot_grad_norm1 = torch.sqrt((qdot_grad1**2)[:,0]+(qdot_grad1**2)[:,1]+(qdot_grad2**2)[:,0]+(qdot_grad2**2)[:,1])
        f_grad_total = qdot_grad_norm1.view(self.nq1,self.nq2).to(device_c).detach()
        
        return f_total,f_grad_total
               
                    
                    
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
        
        
        Xlong = self.qmesh.view(-1,2).to(self.device)
        Xlong.requires_grad=True
        fhtotal_long = self.model.fhatf_forward(Xlong, self.Xstable)
        fh_total = fhtotal_long.view(self.nq1,self.nq2,2).to(device_c).detach()
        co1 = torch.sum(fhtotal_long[:,0])
        co2 = torch.sum(fhtotal_long[:,1])
        qdot_grad1 = grad(co1,Xlong,create_graph=True,retain_graph=True)[0]
        qdot_grad2 = grad(co2,Xlong,create_graph=True,retain_graph=True)[0]
        qdot_grad_norm1 = torch.sqrt((qdot_grad1**2)[:,0]+(qdot_grad1**2)[:,1]+(qdot_grad2**2)[:,0]+(qdot_grad2**2)[:,1])
        fh_grad_total = qdot_grad_norm1.view(self.nq1,self.nq2).to(device_c).detach()
        
        return fg_total,fg_grad_total
        
        
        
        for i in range(self.nq1):
            for j in range(self.nq2):
                X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=self.device, dtype=dtype, requires_grad=True)
                fh_current = self.model.fhat_forward(X)
                self.fh_total[i,j,:] = fh_current
                
    def h_2d(self, X):
        theta = X[:,:2] 
        #b = X[:,3:]
        theta_squared = theta**2
        theta_norm = theta_squared[:,0]+theta_squared[:,1]
        theta_periodic10 =  (torch.sin(theta_norm)/theta_norm*theta[:,0]).view(-1,1)
        theta_periodic11 =  (torch.sin(theta_norm)/theta_norm*theta[:,1]).view(-1,1)
        #theta_periodic12 =  (torch.sin(theta_norm)/theta_norm*theta[:,2]).view(-1,1)
        #theta_periodic1 = torch.cat((theta_periodic10,theta_periodic11,theta_periodic12),dim=1)
        theta_periodic20 = ((1-torch.cos(theta_norm))/theta_norm*(theta[:,0]**2)).view(-1,1)
        theta_periodic21 = ((1-torch.cos(theta_norm))/theta_norm*(theta[:,1]**2)).view(-1,1)
        #theta_periodic22 = ((1-torch.cos(theta_norm))/theta_norm*(theta[:,2]**2)).view(-1,1)
        #theta_periodic2 = torch.cat((theta_periodic20,theta_periodic21,theta_periodic22),dim=1)
        theta_periodic= torch.cat((theta_periodic10,theta_periodic11,
               theta_periodic20,theta_periodic21),dim=1)
        #X_se3 = torch.cat((theta_periodic,b))
        return theta_periodic                
                
    def cal_V(self):
        #V_total = torch.zeros((nq2, nq1), device=device_c, dtype=dtype)
        Xlong = self.qmesh.view(-1,2).to(self.device)+0.1
        Xlong.requires_grad=True
        XXlong = self.h_2d(Xlong)
        Xstable_se3 = self.h_2d(self.Xstable.view(-1,2)).view(4)
        V_long = self.model.V_forward(XXlong, Xstable_se3)
        V_total = V_long.view(self.nq1,self.nq2).to(device_c).detach()
        #for i in range(self.nq1):
        #    for j in range(self.nq2):
        #        X = torch.tensor(([[self.q1[i],self.q2[j]]]), device=self.device, dtype=dtype, requires_grad=True)
        #        V_current = self.model.V_forward(X, self.Xstable)
        #        a = V_current.clone()
        #        V_total[i,j] = a
        print(Xlong[0,:],XXlong[0,:])
        print(V_total)
        return V_total
                
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
        qvalid = self.inverse_kinematics_vec(xlong_valid).to(self.device)
        qvalid.requires_grad=True
        qvalidx = qvalid+torch.tensor([h,0],device=self.device)
        qvalidx_ = qvalid+torch.tensor([-h,0],device=self.device)
        qvalidy = qvalid+torch.tensor([0,h],device=self.device)
        qvalidy_ = qvalid+torch.tensor([0,-h],device=self.device)
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
        ftask = f_task.view(self.nx1,self.nx2,2).detach().numpy()
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
        f_task_grad_valid = torch.sqrt((temp1**2)[:,0]+(temp1**2)[:,1]+(temp2**2)[:,0]+(temp2**2)[:,1])
        f_task_grad = torch.zeros(qlong.shape[0],dtype=dtype)
        f_task_grad[g>0] = f_task_grad_valid
        f_task_grad= f_task_grad.view(self.nx1,self.nx2)
        f_task[g>0] = f_task_valid
        
        return ftask,f_task_grad
        
                
    def plot_f(self,filename, density, widthscale, widthbase, quiver = False, streamplot = True, cmax = 0):

        f_total,f_grad_total = self.cal_f()

        pi = math.pi
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([self.qmin[0],self.qmax[0],self.qmin[1],self.qmax[1]])
        #plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,0].numpy(),self.qtraj[1,0].numpy(),'cs',markersize=15)
        plt.plot(self.qtraj[0,-1].numpy(),self.qtraj[1,-1].numpy(),'bo',markersize=15)
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        fnumpy = f_total.detach()
        U = fnumpy[:,:,0].t().numpy()
        V = fnumpy[:,:,1].t().numpy()
        
        #f streamline
        if(quiver):
            plt.quiver(self.q1_mesh,self.q2_mesh,f_total[:,:,0].detach(),f_total[:,:,1].detach(),color = 'r',width=0.002)
        if(streamplot):
            #print(np.sqrt(U*U+V*V))
            #print(np.ones([self.nq1,self.nq2]))
            #width = np.sqrt(U*U+V*V)*widthscale+widthbase*np.ones(np.shape(U))
            width = 2
            #width[width>10] = 10
            #print(width)
            color = f_grad_total.detach().t().numpy()
            #print(color)
            strm = plt.streamplot(self.q1_mesh.t().numpy(),self.q2_mesh.t().numpy(),fnumpy[:,:,0].t().numpy(),fnumpy[:,:,1].t().numpy(),
                           density = density,color = color,linewidth = width, cmap='autumn')
            plt.colorbar(strm.lines)
            if(cmax):
                plt.clim(0,cmax)
            

        plt.savefig(filename,dpi = 500)
        plt.show()
    
    
    
        
        
    def plot_fhat(self,filename, density, quiver = False, streamplot = True):
        pi = math.pi
        self.cal_fhat()
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([self.qmin[0],self.qmax[0],self.qmin[1],self.qmax[1]])
        #plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        
        fhnumpy = self.fh_total.detach()
        U2 = fhnumpy[:,:,0].t().numpy()
        V2 = fhnumpy[:,:,1].t().numpy()
        
        if(quiver):
            plt.quiver(self.q1_mesh,self.q2_mesh,self.fh_total[:,:,0].detach(),
                       self.fh_total[:,:,1].detach(),color = 'b',width=0.0015)
        if(streamplot):
            plt.streamplot(self.q1_mesh.t().numpy(),self.q2_mesh.t().numpy(),
                           fhnumpy[:,:,0].t().numpy(),fhnumpy[:,:,1].t().numpy(),
                           density = density,color = np.sqrt(U2*U2+V2*V2))
        
        plt.savefig(filename,dpi = 500)
        
        plt.show()
        
    def plot_V(self, filename,num_level):
        pi = math.pi
        V_total = self.cal_V()
            
        
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([self.qmin[0],self.qmax[0],self.qmin[1],self.qmax[1]])
        
        #plt.axis([0, pi, 0, pi])
        widths = 0.001
        #plt.quiver(qtraj[0,:],qtraj[1,:],q_dot[0,:],q_dot[1,:], width=widths)  #trajectory vector field
        plt.plot(self.qtraj[0,0].numpy(),self.qtraj[1,0].numpy(),'cs',markersize=15)
        plt.plot(self.qtraj[0,-1].numpy(),self.qtraj[1,-1].numpy(),'bo',markersize=15)
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        
        Z = V_total.detach()
        plt.contour(self.q1_mesh,self.q2_mesh,Z,
                   levels = np.linspace(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max(), num_level))

        plt.savefig(filename,dpi = 500)


        
    def plot_f_taskspace(self,filename, robot, density, widthscale, widthbase, quiver = False, streamplot = True, cmax = 0):

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
        plt.plot(self.xtraj[0,0].numpy(),self.xtraj[1,0].numpy(),'cs',markersize=15)
        plt.plot(self.xtraj[0,-1].numpy(),self.xtraj[1,-1].numpy(),'bo',markersize=15)
        plt.plot(self.xtraj[0,:].numpy(),self.xtraj[1,:].numpy(),'g')
        #plt.plot((robot.xinit.numpy()[0],robot.xfinal.numpy()[0]),(robot.xinit.numpy()[1],robot.xfinal.numpy()[1]),'g')
        plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])
        
        ftask,f_task_grad = self.cal_ftask()
        
        U = ftask[:,:,0]
        V = ftask[:,:,1]
        
        
                    

        if(quiver):
            plt.quiver(self.x1_mesh.numpy(), self.x2_mesh.numpy(), U, V,'r',width=0.003)
        if(streamplot):
            
            #width = np.sqrt(U*U+V*V)*widthscale+widthbase*np.ones(np.shape(U))
            width = 2
            #width[width>10] = 10
            #print(width)
            color = f_task_grad.detach().t().numpy()
            
            strm = plt.streamplot(self.x1_mesh.t().numpy(), self.x2_mesh.t().numpy(), np.transpose(U), np.transpose(V), density = density ,color =color,linewidth = width,
                                  cmap='autumn')
            plt.colorbar(strm.lines)
            if(cmax):
                plt.clim(0,cmax)
        
        plt.savefig(filename,dpi = 500)
        plt.show()
        
    
    def plot_q_traj(self,filename):

        pi = math.pi
        plt.rcParams["figure.figsize"] = (12.3,10)
        plt.axis([0, pi, 0, pi])
        widths = 0.001
        plt.plot(self.qtraj[0,0].numpy(),self.qtraj[1,0].numpy(),'cs',markersize=15)
        plt.plot(self.qtraj[0,-1].numpy(),self.qtraj[1,-1].numpy(),'bo',markersize=15)
        plt.plot(self.qtraj[0,:].numpy(),self.qtraj[1,:].numpy(),'g') #trajectory
        plt.savefig(filename,dpi = 500)    
        
    def plot_task_traj(self,filename, robot, density, widthscale, widthbase, quiver = False, streamplot = True, cmax = 0):

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
        graphs_axis_length1 = 10*axis_lengths[0]/max_length #+ 0.23*graphs_axis_length2
        #print(graphs_axis_length1,graphs_axis_length2)
        plt.rcParams["figure.figsize"] = (graphs_axis_length1,graphs_axis_length2)
        plt.plot(self.xtraj[0,0].numpy(),self.xtraj[1,0].numpy(),'cs',markersize=15)
        plt.plot(self.xtraj[0,-1].numpy(),self.xtraj[1,-1].numpy(),'bo',markersize=15)
        plt.plot(self.xtraj[0,:].numpy(),self.xtraj[1,:].numpy(),'g')
        #plt.plot((robot.xinit.numpy()[0],robot.xfinal.numpy()[0]),(robot.xinit.numpy()[1],robot.xfinal.numpy()[1]),'g')
        plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])
        plt.savefig(filename,dpi = 500)
        plt.show()
        
    def plot_robot_taskspace(self,filename, robot,theta1,theta2, traj=False):

        l1 = robot.l1
        l2 = robot.l2
        m1 = robot.m1
        m2 = robot.m2
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
        graphs_axis_length1 = 10*axis_lengths[0]/max_length #+ 0.23*graphs_axis_length2
        plt.rcParams["figure.figsize"] = (graphs_axis_length1,graphs_axis_length2)
        if traj == True:
            plt.plot(self.xtraj[0,0].numpy(),self.xtraj[1,0].numpy(),'cs',markersize=15)
            plt.plot(self.xtraj[0,-1].numpy(),self.xtraj[1,-1].numpy(),'bo',markersize=15)
            plt.plot(self.xtraj[0,:].numpy(),self.xtraj[1,:].numpy(),'g')
        plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])

        
        thickness = 10
        linkcolor = [0.8,0.4,0.1]
        joincolor = [0.8,0.1,0.2]
        base = plt.Rectangle((-0.1*(l1+l2), -0.1*(l1+l2)), 0.2*(l1+l2), 0.1*(l1+l2), fill=True, facecolor=joincolor, edgecolor=joincolor, linewidth=2.5)
        plt.plot(0,0,'bo',color=joincolor,markersize=30)
        plt.plot([0,l1*np.cos(theta1)],[0,l1*np.sin(theta1)],linewidth=thickness,color=linkcolor)
        plt.plot(l1*np.cos(theta1),l1*np.sin(theta1),'bo',color='k',markersize=5*m1)
        plt.plot([l1*np.cos(theta1),l1*np.cos(theta1)+l2*np.cos(theta1+theta2)],[l1*np.sin(theta1),l1*np.sin(theta1)+l2*np.sin(theta1+theta2)]
                 ,linewidth=thickness,color=linkcolor)

        plt.plot([l1*np.cos(theta1)+l2*np.cos(theta1+theta2),l1*np.cos(theta1)+(l2*(1+m2*0.01))*np.cos(theta1+theta2)],
                 [l1*np.sin(theta1)+l2*np.sin(theta1+theta2),l1*np.sin(theta1)+(l2*(1+m2*0.01))*np.sin(theta1+theta2)],color='k',linewidth=m2*0.5*thickness)
        

        plt.gca().add_patch(base)
        plt.plot(0,0,'bo',color=[0.2,0.2,0.2],markersize=7)
        plt.plot(l1*np.cos(theta1),l1*np.sin(theta1),'bo',color=[0.2,0.2,0.2],markersize=7)
        #plt.axis([-(l1+l2), (l1+l2), -l2, (l1+l2)])
        plt.savefig(filename,dpi = 500)
        plt.show()