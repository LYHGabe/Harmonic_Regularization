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
import scipy
from scipy.optimize import fsolve
import scipy.interpolate as interpolate
from scipy import integrate



dtype = torch.float
device_c = torch.device("cpu")
device = torch.device("cuda:0")

class robotarm:
    def __init__(self, l1,l2,m1,m2,device):
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
        Kinematic_Riemannian_metric = Kinematic_Riemannian_metric+1.0e-02*torch.eye(2)
        return Kinematic_Riemannian_metric 

    def get_Christoffel_kinematic(self, q):
        G = self.get_Kinematic_Riemannian_metric(q)
        G_inv = torch.inverse(G)
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
    
        
    def qtraj_to_xtraj(self,qtraj):
        xtraj = torch.zeros(qtraj.shape)
        xtraj[0,:] = self.l1*torch.cos(qtraj[0,:])+self.l2*torch.cos(qtraj[0,:]+qtraj[1,:])
        xtraj[1,:] = self.l1*torch.sin(qtraj[0,:])+self.l2*torch.sin(qtraj[0,:]+qtraj[1,:])
        #xinit1 = self.l1*torch.cos(qinit[0])+self.l2*torch.cos(qinit[0]+qinit[1])
        #xinit2 = self.l1*torch.sin(qinit[0])+self.l2*torch.sin(qinit[0]+qinit[1])
        #xfinal1 = self.l1*torch.cos(qfinal[0])+self.l2*torch.cos(qfinal[0]+qfinal[1])
        #xfinal2 = self.l1*torch.sin(qfinal[0])+self.l2*torch.sin(qfinal[0]+qfinal[1])
        #xinit = torch.tensor([[xinit1],[xinit2]],device=self.device, dtype=dtype)
        #xfinal = torch.tensor([[xfinal1],[xfinal2]],device=self.device, dtype=dtype)
        return xtraj
        
        
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

    def Initialize_spline(self, xpoints,ypoints,Vpoints, T, delta, num_timesteps):
        pi = math.pi
        ## aded by YH LEE
        self.qinit = torch.tensor([xpoints[0],ypoints[0]],dtype=dtype)
        self.qfinal = torch.tensor([xpoints[-1],ypoints[-1]],dtype=dtype)
        self.T = T
        self.delta = delta
        self.num_timesteps = num_timesteps
        ##
        rmin = max(0,self.l1-self.l2)
        rmax = self.l1+self.l2
        qmin = [0,0]
        qmax = [pi,pi]
        dt = T/num_timesteps
        qtraj,q_dot,xtraj= self.spline_traj_0309(xpoints,ypoints,Vpoints, T, delta, num_timesteps)
        #qinit,qfinal = self.get_config_points(self.xinit,self.xfinal)
        Xstable = self.qfinal.clone().to(self.device)
        return rmin,rmax,qmin,qmax,dt,self.qinit,self.qfinal,Xstable,xtraj,qtraj,q_dot
    
    def spline_traj_0309(self,xpoints,ypoints,Vpoints, T, delta, num_timesteps):
        af = 10
        deltaT = delta*T
        aa = np.linspace(0,af,xpoints.shape[0])
        tt = np.linspace(0,deltaT,Vpoints.shape[0])
        #a_total = np.linspace(0,af,100)
        tck1 = interpolate.splrep(aa, xpoints)
        tck2 = interpolate.splrep(aa, ypoints)
        tckV = interpolate.splrep(tt, Vpoints)
        
        t_total = np.linspace(0,T,num_timesteps+1)
        timesteps = (t_total[:-1]+t_total[1:])/2
        
        
        #####1. total length calculation
        def splfun(a):
            x = interpolate.splev(a,tck1)
            y = interpolate.splev(a,tck2)
            return x,y
        def splder(a):
            dx = interpolate.splev(a,tck1,der=1)
            dy = interpolate.splev(a,tck2,der=1)
            return dx,dy
        def lenfun(a):
            dx,dy = splder(a)
            dl = np.sqrt(dx**2+dy**2)
            return dl
        def len_int(a):
            length = integrate.quad(lenfun,0,a)[0]
            return length
        
        total_length = len_int(af)
        
        ####2. max_speed calculation
        #max_speed = (total_length/
        #             (T + -1/((1-delta)*(1-delta)*T*T*3)*(T-deltaT)*(T-deltaT)*(T-deltaT)))
        
        def V(t):
            #speed = np.zeros(t.shape)
            
            #speed[t<=deltaT] = interpolate.splev(t[t<=deltaT],tck1)
            if t<=deltaT:
                speed = interpolate.splev(t,tckV)
            #c = Vpoints[-1]/((T-deltaT)**2)
            #speed[t>deltaT] = -c*(t[t>deltaT]-deltaT)+Vpoints[-1]
            else:
                #slope = interpolate.splev(deltaT,tck1,der=1)
                c = Vpoints[-1]/((T-deltaT)**2)
                speed = -c*((t-deltaT)**2)+Vpoints[-1]
            return speed
        
        def l(t):
            #distance = np.zeros(t.shape)
            #for i in range(t.shape):
            #    distance(i)
            distance = integrate.quad(V,0,t)[0]
            #distance = np.zeros(t.shape)
            #distance[t<=deltaT] = max_speed*t[t<=deltaT]
            #distance[t>deltaT] = (max_speed*t[t>deltaT] + -(max_speed)/((1-delta)*(1-delta)*T*T*3)*
            #                      (t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT))
            return distance
        lt_final = l(T)
        V_mul = total_length/lt_final
        def V2(t):
            speed = V_mul*V(t)
            return speed
        def V2mat(t):
            speed = np.zeros(t.shape)
            
            speed[t<=deltaT] = interpolate.splev(t[t<=deltaT],tckV)
            c = Vpoints[-1]/((T-deltaT)**2)
            speed[t>deltaT] = -c*((t[t>deltaT]-deltaT)**2)+Vpoints[-1]
            return V_mul*speed#V_mul*V(t)
        def l2(t):
            distance = integrate.quad(V2,0,t)[0]
            return distance
        #print(timesteps)
        #print(V2mat(timesteps))
        #print(V2(T))
        
        #### 3. matching t vs x,y
        def xy_at_t(t):
            #3-1. t vs l(t)
            lt = l2(t)
            #print(V2(t))
            #print(lt)
            #3-2. l(t) vs a
            def len_res(a): 
                return len_int(a)-lt
            a = fsolve(len_res,(t*af/T))
            x,y = splfun(a)
            return a,x,y
        
        xx = np.zeros([num_timesteps])
        yy = np.zeros([num_timesteps])
        aa = np.zeros([num_timesteps])
        for i in range(num_timesteps):
            #print(timesteps[i])
            aa[i], xx[i],yy[i] = xy_at_t(timesteps[i])
            
        #### 4. asigning speed to the spline
        dx,dy = splder(aa)
        dx_normalized = dx/np.sqrt(dx**2+dy**2)
        dy_normalized = dy/np.sqrt(dx**2+dy**2)
        dx_f = dx_normalized*V2mat(timesteps)
        dy_f = dy_normalized*V2mat(timesteps)
        
        
        #### 5. xtraj and x_dot
        qtraj = torch.zeros(2,num_timesteps)
        q_dot = torch.zeros(2,num_timesteps)
        
        
        
        qtraj[0,:] = torch.from_numpy(xx)
        qtraj[1,:] = torch.from_numpy(yy)
        q_dot[0,:] = torch.from_numpy(dx_f)
        q_dot[1,:] = torch.from_numpy(dy_f)
        xtraj = self.qtraj_to_xtraj(qtraj)
        
        return qtraj,q_dot,xtraj
    
    def spline_traj(self,xpoints,ypoints, T, delta, num_timesteps):
        af = 10
        aa = np.linspace(0,af,xpoints.shape[0])
        #a_total = np.linspace(0,af,100)
        tck1 = interpolate.splrep(aa, xpoints)
        tck2 = interpolate.splrep(aa, ypoints)
        deltaT = delta*T
        t_total = np.linspace(0,T,num_timesteps)
        
        
        #####1. total length calculation
        def splfun(a):
            x = interpolate.splev(a,tck1)
            y = interpolate.splev(a,tck2)
            return x,y
        def splder(a):
            dx = interpolate.splev(a,tck1,der=1)
            dy = interpolate.splev(a,tck2,der=1)
            return dx,dy
        def lenfun(a):
            dx,dy = splder(a)
            dl = np.sqrt(dx**2+dy**2)
            return dl
        def len_int(a):
            length = integrate.quad(lenfun,0,a)[0]
            return length
        
        total_length = len_int(af)
        
        ####2. max_speed calculation
        max_speed = (total_length/
                     (T + -1/((1-delta)*(1-delta)*T*T*3)*(T-deltaT)*(T-deltaT)*(T-deltaT)))
        
        def V(t):
            speed = np.zeros(t.shape)
            speed[t<=deltaT] = max_speed
            speed[t>deltaT] = -(max_speed)/((1-delta)*(1-delta)*T*T)*(t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT) + max_speed
            return speed
        
        def l(t):
            distance = np.zeros(t.shape)
            distance[t<=deltaT] = max_speed*t[t<=deltaT]
            distance[t>deltaT] = (max_speed*t[t>deltaT] + -(max_speed)/((1-delta)*(1-delta)*T*T*3)*
                                  (t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT))
            return distance
        lt = l(t_total)
        
        #### 3. matching t vs x,y
        def xy_at_t(t):
            #3-1. t vs l(t)
            lt = l(t)
            #3-2. l(t) vs a
            def len_res(a): 
                return len_int(a)-lt
            a = fsolve(len_res,(t*af/T))
            x,y = splfun(a)
            return a,x,y
        
        xx = np.zeros([num_timesteps])
        yy = np.zeros([num_timesteps])
        aa = np.zeros([num_timesteps])
        for i in range(num_timesteps):
            aa[i], xx[i],yy[i] = xy_at_t(t_total[i])
            
        #### 4. asigning speed to the spline
        dx,dy = splder(aa)
        dx_normalized = dx/np.sqrt(dx**2+dy**2)
        dy_normalized = dy/np.sqrt(dx**2+dy**2)
        dx_f = dx_normalized*V(t_total)
        dy_f = dy_normalized*V(t_total)
        
        
        #### 5. xtraj and x_dot
        qtraj = torch.zeros(2,num_timesteps)
        q_dot = torch.zeros(2,num_timesteps)
        
        
        
        qtraj[0,:] = torch.from_numpy(xx)
        qtraj[1,:] = torch.from_numpy(yy)
        q_dot[0,:] = torch.from_numpy(dx_f)
        q_dot[1,:] = torch.from_numpy(dy_f)
        xtraj = self.qtraj_to_xtraj(qtraj)
        
        return qtraj,q_dot,xtraj
    
    def spline_traj_old(self,xpoints,ypoints, T, delta, num_timesteps):
        af = 10
        aa = np.linspace(0,af,xpoints.shape[0])
        #a_total = np.linspace(0,af,100)
        tck1 = interpolate.splrep(aa, xpoints)
        tck2 = interpolate.splrep(aa, ypoints)
        deltaT = delta*T
        t_total = np.linspace(0,T,num_timesteps)
        
        
        #####1. total length calculation
        def splfun(a):
            x = interpolate.splev(a,tck1)
            y = interpolate.splev(a,tck2)
            return x,y
        def splder(a):
            dx = interpolate.splev(a,tck1,der=1)
            dy = interpolate.splev(a,tck2,der=1)
            return dx,dy
        def lenfun(a):
            dx,dy = splder(a)
            dl = np.sqrt(dx**2+dy**2)
            return dl
        def len_int(a):
            length = integrate.quad(lenfun,0,a)[0]
            return length
        
        total_length = len_int(af)
        
        ####2. max_speed calculation
        max_speed = (total_length/
                     (T + -1/((1-delta)*(1-delta)*T*T*3)*(T-deltaT)*(T-deltaT)*(T-deltaT)))
        
        def V(t):
            speed = np.zeros(t.shape)
            speed[t<=deltaT] = max_speed
            speed[t>deltaT] = -(max_speed)/((1-delta)*(1-delta)*T*T)*(t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT) + max_speed
            return speed
        
        def l(t):
            distance = np.zeros(t.shape)
            distance[t<=deltaT] = max_speed*t[t<=deltaT]
            distance[t>deltaT] = (max_speed*t[t>deltaT] + -(max_speed)/((1-delta)*(1-delta)*T*T*3)*
                                  (t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT))
            return distance
        lt = l(t_total)
        
        #### 3. matching t vs x,y
        def xy_at_t(t):
            #3-1. t vs l(t)
            lt = l(t)
            #3-2. l(t) vs a
            def len_res(a): 
                return len_int(a)-lt
            a = fsolve(len_res,(t*af/T))
            x,y = splfun(a)
            return a,x,y
        
        xx = np.zeros([num_timesteps])
        yy = np.zeros([num_timesteps])
        aa = np.zeros([num_timesteps])
        for i in range(num_timesteps):
            aa[i], xx[i],yy[i] = xy_at_t(t_total[i])
            
        #### 4. asigning speed to the spline
        dx,dy = splder(aa)
        dx_normalized = dx/np.sqrt(dx**2+dy**2)
        dy_normalized = dy/np.sqrt(dx**2+dy**2)
        dx_f = dx_normalized*V(t_total)
        dy_f = dy_normalized*V(t_total)
        
        
        #### 5. xtraj and x_dot
        xtraj = torch.zeros(2,num_timesteps)
        x_dot = torch.zeros(2,num_timesteps)
        
        xtraj[0,:] = torch.from_numpy(xx)
        xtraj[1,:] = torch.from_numpy(yy)
        x_dot[0,:] = torch.from_numpy(dx_f)
        x_dot[1,:] = torch.from_numpy(dy_f)
        
        return xtraj,x_dot
        
        