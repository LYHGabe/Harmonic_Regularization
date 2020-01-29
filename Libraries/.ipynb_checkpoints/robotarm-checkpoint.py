import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.deterministic=True
from torch.nn import functional as F
import torch.optim as optim
import math

dtype = torch.float
device_c = torch.device("cpu")
device = torch.device("cuda:0")

class robotarm:
    def __init__(self, l1,l2,m1,m2,device=device):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.device = device
    def forward_kinematics(self, q):
        x = np.zeros(2)
        x[0] = self.l1*np.cos(q[0])+self.l2*np.cos(q[0]+q[1])
        x[1] = self.l1*np.sin(q[0])+self.l2*np.sin(q[0]+q[1])
        return x    
    
    def inverse_kinematics(self,x):
        x1 = x[0]
        x2 = x[1]
        q2 = np.arccos((x1**2+x2**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2))
        q1 = np.arccos(((self.l1+self.l2*np.cos(q2))*x1+(self.l2*np.sin(q2))*x2)/(self.l1**2+self.l2**2+2*self.l1*self.l2*np.cos(q2)))
        return np.vstack([q1,q2])
    
    def get_jacobian_link1(self, q):
        Jacobian = torch.zeros(2,2)
        Jacobian = torch.tensor([[-self.l1*torch.sin(q[0]), 0],[self.l1*torch.cos(q[0]), 0]])
        return Jacobian
        
        
    def get_jacobian(self, q):
        Jacobian = torch.zeros(2,2)
        Jacobian[0,0] = -self.l1*torch.sin(q[0])-self.l2*torch.sin(q[0]+q[1])
        Jacobian[1,0] = self.l1*torch.cos(q[0])+self.l2*torch.cos(q[0]+q[1])
        Jacobian[0,1] = -self.l2*torch.sin(q[0]+q[1])
        Jacobian[1,1] = self.l2*torch.cos(q[0]+q[1])
        return Jacobian
    
    def get_Kinematic_Riemannian_metric(self, q):
        Kinematic_Riemannian_metric = torch.tensor([[0,0],[0,0]], dtype=torch.float32)
        Jacobian = self.get_jacobian(q)
        Jacobian_transpose = Jacobian.t()
        Kinematic_Riemannian_metric = torch.mm(Jacobian_transpose,Jacobian)
        Kinematic_Riemannian_metric = Kinematic_Riemannian_metric#+1.0e-03*torch.eye(2)
        return Kinematic_Riemannian_metric 

    def get_Christoffel_kinematic(self, q):
        G = self.get_Kinematic_Riemannian_metric(q)
        G_inv = torch.inverse(G+1.0e-03*torch.eye(2))
        Gamma_1 = 0.5*G_inv.matmul(torch.tensor([[0, -2*self.l1*self.l2*torch.sin(q[1])],
                                                 [2*self.l1*self.l2*torch.sin(q[1]), 0]]))
        Gamma_2 = 0.5*G_inv.matmul(torch.tensor([[-2*self.l1*self.l2*torch.sin(q[1]), -2*self.l1*self.l2*torch.sin(q[1])],
                                                 [0, 0]]))
        return Gamma_1, Gamma_2
    
    def get_Kinetic_Riemannian_metric(self, q):
        Mass = torch.tensor([[0,0],[0,0]])
        Jacobian1 = self.get_jacobian_link1(q)
        Jacobian1_transpose = torch.t(Jacobian1)
        Jacobian2 = self.get_jacobian(q)
        Jacobian2_transpose = torch.t(Jacobian2)     
        Mass = self.m1*Jacobian1_transpose.matmul(Jacobian1) + self.m2*Jacobian2_transpose.matmul(Jacobian2)
        return Mass
    
    def get_Christoffel_kinetic(self, q):
        G = self.get_Kinetic_Riemannian_metric(q)
        G_inv = torch.inverse(G)
        
        Gamma_1 = self.m2*0.5*G_inv.matmul(torch.tensor([[0, -2*self.l1*self.l2*torch.sin(q[1])],[2*self.l1*self.l2*torch.sin(q[1]), 0]]))
        Gamma_2 = self.m2*0.5*G_inv.matmul(torch.tensor([[-2*self.l1*self.l2*torch.sin(q[1]), -2*self.l1*self.l2*torch.sin(q[1])],[0, 0]]))
        
        return Gamma_1, Gamma_2   
    
    def get_x_points(self, qinit, qfinal):
        xinit1 = self.l1*torch.cos(qinit[0])+self.l2*torch.cos(qinit[0]+qinit[1])
        xinit2 = self.l1*torch.sin(qinit[0])+self.l2*torch.sin(qinit[0]+qinit[1])
        xfinal1 = self.l1*torch.cos(qfinal[0])+self.l2*torch.cos(qfinal[0]+qfinal[1])
        xfinal2 = self.l1*torch.sin(qfinal[0])+self.l2*torch.sin(qfinal[0]+qfinal[1])
        xinit = torch.tensor([[xinit1],[xinit2]],device=self.device, dtype=dtype)
        xfinal = torch.tensor([[xfinal1],[xfinal2]],device=self.device, dtype=dtype)
        return xinit, xfinal
    
    def get_config_points(self, xinit,xfinal):
        qinit  = torch.zeros(xinit.shape)
        qfinal = torch.zeros(xfinal.shape)
        xi1 = xinit[0]
        xi2 = xinit[1]
        xf1 = xfinal[0]
        xf2 = xfinal[1]
        
        Ci2 = (xi1*xi1+xi2*xi2-self.l1*self.l1-self.l2*self.l2)/(2*self.l1*self.l2)
        #print(C2)
        if (Ci2 < -1 or Ci2 > 1):
            print( 'Initial point is not feasible. Please try another conbination')
            #return q
        Si2 = torch.sqrt(1-Ci2*Ci2)
        qi1 = torch.atan2(xi2,xi1)-torch.atan2((self.l2*Si2),(self.l1+self.l2*Ci2))
        qi2 = torch.acos(Ci2)
        qinit = torch.tensor([qi1, qi2])
        
        Cf2 = (xf1*xf1+xf2*xf2-self.l1*self.l1-self.l2*self.l2)/(2*self.l1*self.l2)
        #print(C2)
        if (Cf2 < -1 or Cf2 > 1):
            print( 'Final point is not feasible. Please try another conbination')
            #return q
        Sf2 = torch.sqrt(1-Cf2*Cf2)
        qf1 = torch.atan2(xf2,xf1)-torch.atan2((self.l2*Sf2),(self.l1+self.l2*Cf2))
        qf2 = torch.acos(Cf2)
        qfinal = torch.tensor([qf1, qf2])
        
        return qinit, qfinal
        
        
        
        for i in range(n2):
            x1_current = xtraj[0,i]
            x2_current = xtraj[1,i]
            C2 = (x1_current*x1_current+x2_current*x2_current-self.l1*self.l1-self.l2*self.l2)/(2*self.l1*self.l2)
            #print(C2)
            if (C2 < -1 or C2 > 1):
                print( 'Point #'+str(i+1) + ' is not feasible. Please try another conbination')
                #return q
            S2 = torch.sqrt(1-C2*C2)
            q1_current = torch.atan2(x2_current,x1_current)-torch.atan2((self.l2*S2),(self.l1+self.l2*C2))
            q2_current = torch.acos(C2)
            qtraj[:,i] = torch.tensor([q1_current, q2_current])
            #print(torch.cos(q2_current)-C2)
        return qtraj
    
    def task_trajectory(self, xinit, xfinal, T, delta, num_timesteps):
        #length = torch.dot(xfinal-xinit,xfinal-xinit)
        #print(xinit.data[0,0])
        #max_speed = 1
        timesteps = torch.linspace(0,T,num_timesteps+1)
        x_dot = torch.zeros(2,num_timesteps)
        xtraj = torch.zeros(2,num_timesteps)
        deltaT = delta*T
        direction = (xfinal-xinit)/torch.norm((xfinal-xinit), 2)
        max_speed = (torch.norm((xfinal-xinit), 2)/
                     (T + -1/((1-delta)*(1-delta)*T*T*3)*(T-deltaT)*(T-deltaT)*(T-deltaT)))
        
        def V(t):
            if t<=deltaT:
                speed = max_speed
            else:
                speed = -(max_speed)/((1-delta)*(1-delta)*T*T)*(t-deltaT)*(t-deltaT) + max_speed
            return speed
        
        def s(t):
            if t<=deltaT:
                distance = max_speed*t
            else:
                distance = (max_speed*t + -(max_speed)/((1-delta)*(1-delta)*T*T*3)*
                            (t-deltaT)*(t-deltaT)*(t-deltaT))
            return distance
        
        #print( direction)
        for i in range(num_timesteps):
            #print(i)
            t = (timesteps[i]+timesteps[i+1])/2
            vel_current = V(t)
            dist_current = s(t)
            x_dot[0,i] = vel_current*direction[0]
            x_dot[1,i] = vel_current*direction[1]
            #print(xtraj)
            
            xtraj[0,i] = xinit[0]+dist_current*direction[0]
            xtraj[1,i] = xinit[1]+dist_current*direction[1]
        
        return xtraj, x_dot
    
    def task_trajectory_complex(self, ctx, T, delta, num_timesteps):
        #length = torch.dot(xfinal-xinit,xfinal-xinit)
        #print(xinit.data[0,0])
        #max_speed = 1
        timesteps = torch.linspace(0,T,num_timesteps+1)
        x_dot = torch.zeros(2,num_timesteps)
        xtraj = torch.zeros(2,num_timesteps)
        deltaT = delta*T
        direction = (xfinal-xinit)/torch.norm((xfinal-xinit), 2)
        max_speed = (torch.norm((xfinal-xinit), 2)/
                     (T + -1/((1-delta)*(1-delta)*T*T*3)*(T-deltaT)*(T-deltaT)*(T-deltaT)))
        
        def V(t):
            if t<=deltaT:
                speed = max_speed
            else:
                speed = -(max_speed)/((1-delta)*(1-delta)*T*T)*(t-deltaT)*(t-deltaT) + max_speed
            return speed
        
        def s(t):
            if t<=deltaT:
                distance = max_speed*t
            else:
                distance = (max_speed*t + -(max_speed)/((1-delta)*(1-delta)*T*T*3)*
                            (t-deltaT)*(t-deltaT)*(t-deltaT))
            return distance
        
        #print( direction)
        for i in range(num_timesteps):
            #print(i)
            t = (timesteps[i]+timesteps[i+1])/2
            vel_current = V(t)
            dist_current = s(t)
            x_dot[0,i] = vel_current*direction[0]
            x_dot[1,i] = vel_current*direction[1]
            #print(xtraj)
            
            xtraj[0,i] = xinit[0]+dist_current*direction[0]
            xtraj[1,i] = xinit[1]+dist_current*direction[1]
        
        return xtraj, x_dot
        
        
    def config_trajectory(self, xtraj):
        qtraj = torch.zeros(xtraj.shape)
        n1 = qtraj.shape[0]
        n2 = qtraj.shape[1]
        for i in range(n2):
            x1_current = xtraj[0,i]
            x2_current = xtraj[1,i]
            C2 = (x1_current*x1_current+x2_current*x2_current-self.l1*self.l1-self.l2*self.l2)/(2*self.l1*self.l2)
            #print(C2)
            if (C2 < -1 or C2 > 1):
                print( 'Point #'+str(i+1) + ' is not feasible. Please try another conbination')
                #return q
            S2 = torch.sqrt(1-C2*C2)
            q1_current = torch.atan2(x2_current,x1_current)-torch.atan2((self.l2*S2),(self.l1+self.l2*C2))
            q2_current = torch.acos(C2)
            qtraj[:,i] = torch.tensor([q1_current, q2_current])
            #print(torch.cos(q2_current)-C2)
            #qtraj = qtraj.t().to(self.device)
        return qtraj
    
    
    def config_velocity(self, qtraj, x_dot):
        n1 = qtraj.shape[0]
        n2 = qtraj.shape[1]
        q_dot = torch.zeros(n1,n2)
        for i in range(n2):
            q = torch.tensor([qtraj[0,i],qtraj[1,i]])
            if q[1] == 0:
                print('Singularity! q2 should not be zero.')
                return
            Jacobian = self.get_jacobian(q)
            #print(q)
            #print(Jacobian)
            inv_J = torch.inverse(Jacobian)
            x_dot_current = torch.tensor([[x_dot[0,i]],[x_dot[1,i]]])
            q_dot_current = torch.matmul(inv_J,x_dot_current)
            q_dot_current = inv_J.matmul(x_dot_current)
            
            q_dot[:,i] = torch.t(q_dot_current)
            #q_dot = q_dot.t().to(self.device)
        return q_dot
    
    def Initialize(self, xinit,xfinal,T,delta,num_timesteps):
        pi = math.pi
        
        ## aded by YH LEE
        self.xinit = xinit
        self.xfinal = xfinal
        self.T = T
        self.delta = delta
        self.num_timesteps = num_timesteps
        ##
        rmin = max(0,self.l1-self.l2)
        rmax = self.l1+self.l2
        qmin = [0,0]
        qmax = [pi,pi]
        dt = T/num_timesteps
        qinit,qfinal = self.get_config_points(xinit,xfinal)
        Xstable = qfinal.clone().to(self.device)        
        [xtraj,x_dot] = self.task_trajectory(xinit, xfinal, T, delta, num_timesteps)
        qtraj = self.config_trajectory(xtraj)
        q_dot = self.config_velocity(qtraj,x_dot)
        #print(rmin,rmax)
        return rmin,rmax,qmin,qmax,dt,qinit,qfinal,Xstable,xtraj,x_dot,qtraj,q_dot
        
        
