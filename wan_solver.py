#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wan_solver.py
@Time    :   2023/09/07 16:20:26
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import numpy as np
import time
import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MLPBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.mlp(x)
       # x = torch.sin(x)
        return x

class wan_pde_solver(nn.Module):
    def __init__(self, N_dm, N_bd, beta, v_step, u_step,
                v_lr=0.001, u_lr=0.001, dim=2, x_range=[-1,1], t_range=[0,1],
                 iteration=20000):
        super().__init__()
        
        self.x_l, self.x_r = x_range[0], x_range[1]         # x range [-1, 1] * [-1, 1]
        self.t0, self.t1 = t_range[0], t_range[1]           # t range [0, 1]
        self.dim= dim                           #dimension of the problem
        self.dm_size= N_dm                      #collocation points in domain                   
        self.bd_size= N_bd                      #collocation points on domain boundary

        self.iteration= iteration

        self.beta= beta                         # boundary condition hyperparameters
                    
        self.v_step= v_step                       
        self.v_lr = v_lr  
        #                      
        self.u_step= u_step              
        self.u_lr = u_lr

        ## integral x,t
        self.int_x = (self.x_r-self.x_l)**2
        self.int_t = self.t1-self.t0
        # Neural Net for trial function
        self.u_net = nn.Sequential(
            MLPBlock(self.dim+1, 32, 32),
            MLPBlock(32, 32, 16),
            nn.Linear(16,1),
        )
        # Neural Net for test function
        self.v_net = nn.Sequential(
            MLPBlock(self.dim+1, 16, 32),
            MLPBlock(32, 32, 16),
            nn.Linear(16,1),
        )
    
        self.u_opt = torch.optim.Adam(self.v_net.parameters(), lr=self.u_lr)
        self.v_opt = torch.optim.Adam(self.u_net.parameters(), lr=self.v_lr)

    def sample_train(self, dm_size, bd_size, dim=2):
        '''
        Equation: du/dt - Laplace u = f
        u(x,t) = g(x,t)= 2*sin(pi/2*x1)cos(pi/2*x2)*exp{-t}
        Thus, h(x)= 2*sin(pi/2*x1)cos(pi/2*x2)
        f(x,t) = (pi**2-2)sin(pi/2*x1)cos(pi/2*x2)*exp{-t}
        '''       
        # colloaction points in the domain
        x_dm = torch.distributions.uniform.Uniform(self.x_l,self.x_r).sample([dm_size,dim])
        t_dm = torch.distributions.uniform.Uniform(self.t0, self.t1).sample([dm_size,1])
        xt_dm = torch.cat((x_dm, t_dm), dim=1)
        # value of u, f , dim: [x_inside_size,x_inside_size, t_size]
        u_true = 2*torch.sin(torch.pi/2.*xt_dm[:,0])*torch.cos(torch.pi/2.*xt_dm[:,1])*torch.exp(-xt_dm[:,-1])
        f_obv = (torch.pi**2/2.-1.) * u_true

        ## initial condition h(x)
        #x_init = torch.distributions.uniform.Uniform(self.x_l,self.x_r).sample([dm_size,dim])
        x_init = torch.cat((x_dm, torch.ones(dm_size, 1)* self.t0),dim=1)
        #x_end = torch.distributions.uniform.Uniform(self.x_l,self.x_r).sample([dm_size,dim])
        x_end = torch.cat((x_dm, torch.ones(dm_size, 1)* self.t1),dim=1)

        # value of h
        h_obv = 2*torch.sin(torch.pi/2.*x_init[:,0])*torch.cos(torch.pi/2.*x_init[:,1])
        ## boundary condition g(x,t)
        x_bd_list = []
        for i in range(dim):
            x_bd = torch.distributions.uniform.Uniform(self.x_l,self.x_r).sample([bd_size,dim])
            t_bd = torch.distributions.uniform.Uniform(self.t0,self.t1).sample([bd_size,1])
            x_bd[:,i] = self.x_r
            x_bd_list.append(torch.cat((x_bd,t_bd),dim=1))
            x_bd = torch.distributions.uniform.Uniform(self.x_l,self.x_r).sample([bd_size,dim])
            t_bd = torch.distributions.uniform.Uniform(self.t0,self.t1).sample([bd_size,1])
            x_bd[:,i] = self.x_l
            x_bd_list.append(torch.cat((x_bd,t_bd),dim=1))
        x_bd = torch.cat(x_bd_list,dim=0)

        ## boundary of u
        bd_obv = 2*torch.sin(torch.pi/2.*x_bd[:,0])*torch.cos(torch.pi/2.*x_bd[:,1])*torch.exp(-x_bd[:,-1])

        return (xt_dm, x_init, x_end, x_bd, f_obv.view(-1,1), h_obv.view(-1,1), bd_obv.view(-1,1))

    def sample_test(self, mesh_size, time_size, test_size, dim=2):
        '''
        Equation: du/dt - Laplace u = f
        u(x,t) = g(x,t)= 2*sin(pi/2*x1)cos(pi/2*x2)*exp{-t}
        Thus, h(x)= 2*sin(pi/2*x1)cos(pi/2*x2)
        f(x,t) = (pi**2-2)sin(pi/2*x1)cos(pi/2*x2)
        '''       
        # make data in the domain
        x_mesh = torch.linspace(self.x_l,self.x_r,mesh_size)
        t_mesh = torch.linspace(self.t0,self.t1,time_size)
        x1, x2, t = torch.meshgrid(x_mesh, x_mesh, t_mesh,indexing="ij")
        x1 = x1.reshape(-1)
        x2 = x2.reshape(-1)
        t = t.reshape(-1)
        ## value of x, dim: [mesh_size*mesh_size*time_size,dim+1]
        x = torch.stack((x1,x2,t),dim=1)
        ## value of u (x,t), dim: [mesh_size*mesh_size*time_size]
        u_true = 2*torch.sin(torch.pi/2.*x[:,0])*torch.cos(torch.pi/2.*x[:,1])*torch.exp(-x[:,-1])  
        ## random test data
        # x_inside = torch.distributions.uniform.Uniform(self.x_l,self.x_r).sample([test_size,dim])
        # t_inside = torch.distributions.uniform.Uniform(self.t0, self.t1).sample([test_size,1])
        # x_inside = torch.cat((x_inside, t_inside), dim=1)
        # # value of u (x,t), dim: [test_size, test_size, test_size]
        # u_true = 2*torch.sin(torch.pi/2.*x_inside[:,0])*torch.cos(torch.pi/2.*x_inside[:,1])*torch.exp(-x_inside[:,-1])        
        return (x, u_true.view(-1,1))

    def grad(self, y, x, create_graph=True, keepdim=False):
        '''
        y: [N, Ny] or [Ny]
        x: [N, Nx] or [Nx]
        Return dy/dx ([N, Ny, Nx] or [Ny, Nx]).
        '''
        N = y.size(0) if len(y.size()) == 2 else 1
        Ny = y.size(-1)
        Nx = x.size(-1)
        z = torch.ones_like(y[..., 0])
        dy = []
        for i in range(Ny):
            dy.append(torch.autograd.grad(y[..., i], x, grad_outputs=z, create_graph=create_graph)[0])
        shape = np.array([N, Ny])[2-len(y.size()):]
        shape = list(shape) if keepdim else list(shape[shape > 1])
        return torch.cat(dy, dim=-1).view(shape + [Nx])

    def forward_u(self, x_in):
        x_in.requires_grad_(True)
        ## u(x, t)
        u_val = self.u_net(x_in)
        ## grad_u(x, t)
        grad_u = self.grad(u_val, x_in)
        du_x, du_y, du_t = grad_u[:,0].view(-1,1), grad_u[:,1].view(-1,1), grad_u[:,-1].view(-1,1)
        return (u_val, du_x, du_y, du_t)
        
    def forward_v(self, x_in):
        x_in.requires_grad_(True)
        ## v(x, t)
        v_val= self.v_net(x_in)
        ## grad_v(x, t)
        grad_v = self.grad(v_val, x_in)
        dv_x, dv_y, dv_t = grad_v[:,0].view(-1,1), grad_v[:,1].view(-1,1), grad_v[:,-1].view(-1,1)
        return (v_val, dv_x, dv_y, dv_t)
    
    def constraints_w(self, x_in, diff=False):
        # decay function w = e^(-1/(x^2-1))/I  Boundary tends to 0
        # so test function with constraints: p = w*v
        I1 = 0.210987 #decay_coefficient
        
        w = torch.exp(-1./((x_in[:,0]**2-1)*(x_in[:,1]**2-1)))/I1
        w = torch.where(torch.isinf(w), torch.full_like(w,0), w)
        if diff:
            # dwx = 2x/(I(x**2-1)**2) * w
            dw_x = 2.* x_in[:,0] / ((x_in[:,0]**2 -1 )**2 * (x_in[:,1]**2-1)) * w /I1
            dw_y = 2.* x_in[:,1] / ((x_in[:,1]**2 -1 )**2 * (x_in[:,0]**2-1)) * w /I1
            dw_x = torch.where(torch.isnan(dw_x), torch.full_like(dw_x,0),dw_x)
            dw_y = torch.where(torch.isnan(dw_y), torch.full_like(dw_y,0),dw_y)
            return w.view(-1,1), dw_x.view(-1,1), dw_y.view(-1,1)
        else:
            return w.view(-1,1)

    def build(self, x_dm, x_init, x_end, x_bd, f_obv, h_obv, bd_obv):
        
        ## initial condition u(0), v(0), p(0)
        u_init = self.u_net(x_init)
        v_init = self.v_net(x_init)
        w_init = self.constraints_w(x_init[:,:-1])
        p_init = w_init * v_init


        ## end time u(T), v(T), p(T)
        u_end = self.u_net(x_end)
        v_end = self.v_net(x_end)
        w_end = self.constraints_w(x_end[:,:-1])
        p_end = w_end * v_end

        ## boundary condition
        u_bd = self.u_net(x_bd)
        ## inside domain
        u_dm, dux_dm, duy_dm, dut_dm = self.forward_u(x_dm)
        v_dm, dvx_dm, dvy_dm, dvt_dm = self.forward_v(x_dm)
        w_dm, dwx_dm, dwy_dm = self.constraints_w(x_dm[:,:-1],diff=True)
        p_dm = w_dm * v_dm

        ## nabla u * nabla p = nabla u * (v nabla w + w nabla v)
        dux_dp_dm = dux_dm * (v_dm * dwx_dm + w_dm * dvx_dm)
        duy_dp_dm = duy_dm * (v_dm * dwy_dm + w_dm * dvy_dm)
        ## u * dp/dt = u * (w * dv/dt + v * dw/dt)
        u_dpt = u_dm * w_dm * dvt_dm
        ## f * p = f * w * v
        f_p_dm = f_obv * w_dm * v_dm
        ## volume of the whole domain
        int_dm = self.int_x * self.int_t

        test_norm = torch.mean(p_dm**2) * int_dm
        ## integral term: 
        ## 
        ## 1. int du/dt * p = u * p - int dv/dt * u = u(T)*p(T) - h*p(0) - int dp/dt * u
        int_l1 = torch.mean((u_end * p_end - h_obv * p_init)) * self.int_x
        int_l2 = torch.mean(u_dpt) * int_dm
        ## 2. int nabla u * nabla v
        int_l3 = torch.mean((dux_dp_dm + duy_dp_dm)) * int_dm
        ## 3. int f * p
        int_l4 = torch.mean(f_p_dm) * int_dm

        loss_int = torch.square(int_l1 - int_l2 + int_l3 - int_l4) / test_norm     
        
        loss_bd = torch.norm(u_bd-bd_obv) + torch.norm(u_init-h_obv)

        loss_u = self.beta * loss_bd + loss_int
        loss_v = -torch.log(loss_int)
        return loss_u, loss_v, loss_int, loss_bd

    def build2(self, x_dm, x_init, x_end, x_bd, f_obv, h_obv, bd_obv):
        
        ## initial condition u(0), v(0), p(0)
        u_init = self.u_net(x_init)
        v_init = self.v_net(x_init)
        w_init = self.constraints_w(x_init[:,:-1])
        p_init = w_init * v_init


        ## end time u(T), v(T), p(T)
        u_end = self.u_net(x_end)
        v_end = self.v_net(x_end)
        w_end = self.constraints_w(x_end[:,:-1])
        p_end = w_end * v_end

        ## boundary condition
        u_bd = self.u_net(x_bd)
        ## inside domain
        u_dm, dux_dm, duy_dm, dut_dm = self.forward_u(x_dm)
        v_dm, dvx_dm, dvy_dm, dvt_dm = self.forward_v(x_dm)
        w_dm, dwx_dm, dwy_dm = self.constraints_w(x_dm[:,:-1],diff=True)
        p_dm = w_dm * v_dm

        ## nabla u * nabla p = nabla u * (v nabla w + w nabla v)
        dux_dp_dm = dux_dm * (v_dm * dwx_dm + w_dm * dvx_dm)
        duy_dp_dm = duy_dm * (v_dm * dwy_dm + w_dm * dvy_dm)
        ## u * dp/dt = u * (w * dv/dt + v * dw/dt)
        u_dpt = u_dm * w_dm * dvt_dm
        ## f * p = f * w * v
        f_p_dm = f_obv * w_dm * v_dm
        ## volume of the whole domain
        int_dm = self.int_x * self.int_t

        test_norm = torch.mean(p_dm**2) * int_dm
        ## integral term: 
        ## 
        ## 1. int du/dt * p = u * p - int dv/dt * u = u(T)*p(T) - h*p(0) - int dp/dt * u
        int_l1 = (u_end * p_end - h_obv * p_init) * self.int_x
        int_l2 = u_dpt * int_dm
        ## 2. int nabla u * nabla v
        int_l3 = (dux_dp_dm + duy_dp_dm) * int_dm
        ## 3. int f * p
        int_l4 = f_p_dm * int_dm

        loss_int = torch.norm(int_l1 - int_l2 + int_l3 - int_l4)     
        
        loss_bd = torch.norm(u_bd-bd_obv) + torch.norm(u_init-h_obv)

        loss_u = self.beta * loss_bd + loss_int
        loss_v = loss_int
        return loss_u, loss_v, loss_int, loss_bd
    

    def forward(self, x):
        u = self.u_net(x)
        return u

    def main_fun(self):
        #*********************************************************************
 
        #*********************************************************************
        # generate points for testing usage
        test_data= self.sample_test(20, 20, self.dim)
        x_test, u_test = test_data

        #
        start_time = time.time()
        for itr in range(self.iteration):
            #print("********** Iteration {} ************".format(itr))
            #print("time elapsed: {:.2f} s".format(time.time() - start_time))
            ## sampling step ##
            sample_start = time.time()
            train_data= self.sample_train(self.dm_size, self.bd_size, self.dim)

            x_dm, x_init, x_end, x_bd, f_obv, h_obv, bd_obv = train_data
            sample_time = time.time()-sample_start
            #print("{:.2f} s to sample".format(sample_time))

            ## training step ##
            optimizer_start = time.time()
            

            # alternating training
            for _ in range(self.v_step):
                loss_u, loss_v, loss_int, loss_bd = self.build2(x_dm, x_init, x_end, x_bd, f_obv, h_obv, bd_obv)
                self.v_opt.zero_grad()
                loss_v.backward()
                self.v_opt.step()
            for _ in range(self.u_step):
                loss_u, loss_v, loss_int, loss_bd = self.build2(x_dm, x_init, x_end, x_bd, f_obv, h_obv, bd_obv)
                self.u_opt.zero_grad()
                loss_u.backward()
                self.u_opt.step()
            
            opt_time = time.time() - optimizer_start
            #print("{:.2f} s to optimizer| loss_u {:6.3f}, loss_v {:6.3f}.".format(opt_time, loss_u, loss_v))
            ## test step ##
            if itr % 50 ==0:
                pred_u = self.forward(x_test)
                err_l2= torch.norm(pred_u-u_test)
                print("iteration {:n}: loss_u:{:6.3f} loss_v:{:6.3f} loss_int:{:6.3f} loss_bd:{:6.3f} test_norm:{:6.3f}".format(itr, loss_u, loss_v, loss_int, loss_bd, err_l2))

    
if __name__ == '__main__':
    solver = wan_pde_solver(N_dm = 40000,
                            N_bd = 200,
                            beta = 2000,
                            v_step = 1,
                            u_step = 2,
                            v_lr = 0.04,
                            u_lr = 0.015)
    solver.main_fun()
    pass
