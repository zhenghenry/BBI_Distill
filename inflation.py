from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch

from pdb import set_trace as bp

class BBI(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate
    deltaV (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): exponent in the Hamiltonian H=Pi^2 V^eta
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): L2 regularization coefficient
    nu (float): parameter that controls the size of the bounces at each step
    gamma (float): parameter that damps the bounces when close to the minimum
    '''

    def __init__(self, params, lr=required, deltaV=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=0.001, gamma=0., weight_decay=0, eta=1., plus1=False, consEn=True, debug=False):
        defaults = dict(lr=lr, deltaV=deltaV, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, gamma=gamma, weight_decay=weight_decay, eta=eta, plus1=plus1, consEn=consEn, debug=debug)
        self.deltaV = deltaV
        self.deltaEn = deltaEn
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.debug = debug
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.gamma = gamma
        self.plus1 = plus1
        super(BBI, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.deltaV)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(1*self.plus1+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor((1.-self.plus1), device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)

        if V > self.eps2:

            if self.debug :
                print('Relative energy violation:', (torch.log(1*self.plus1+self.p2)+torch.log(V)-self.energy)/self.energy)

            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                p2true = ((self.expenergy / V)-1*self.plus1)
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
                    if self.debug :
                        print('done')

            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.deltaV))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(1.*self.plus1+self.p2))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + ((V*(self.deltaEn+1.*self.plus1)/self.expenergy)**self.gamma)*self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                   
            self.p2 = pnorm**2 

            self.iteration += 1

        return loss




class RLSepGen(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate
    deltaV (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): exponent in the Hamiltonian H=Pi^2 V^eta
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): L2 regularization coefficient
    nu (float): parameter that controls the size of the bounces at each step
    gamma (float): parameter that damps the bounces when close to the minimum
    '''

    def __init__(self, params, lr=required, deltaV=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=0.001, gamma=0., weight_decay=0, eta=1., consEn=True, debug=False):
        defaults = dict(lr=lr, deltaV=deltaV, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, gamma=gamma, weight_decay=weight_decay, eta=eta, consEn=consEn, debug=debug)
        self.deltaV = deltaV
        self.deltaEn = deltaEn
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.debug = debug
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.gamma = gamma
        super(RLSepGen, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.deltaV)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(1+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)

        if V > self.eps2:

            if self.debug :
                print('Relative energy violation:', (torch.log(1+self.p2)+torch.log(V)-self.energy)/self.energy)

            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                p2true = ((self.expenergy / V)-1)
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
                    if self.debug :
                        print('done')

            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.deltaV))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(1+self.p2))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + ((V*(self.deltaEn+1)/self.expenergy)**self.gamma)*self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                   
            self.p2 = pnorm**2 

            self.iteration += 1

        return loss


class RLSepGen2nd(Optimizer):
    '''
    Second order optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate
    deltaV (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): exponent in the Hamiltonian H=Pi^2 V^eta
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): L2 regularization coefficient
    nu (float): parameter that controls the size of the bounces at each step
    gamma (float): parameter that damps the bounces when close to the minimum

    '''
    def __init__(self, params, lr=required, deltaV=0., eps1=1e-10, eps2=1e-40, deltaEn=0., weight_decay=0, eta=1., consEn=True, nu=0.001, gamma=1.0, debug=False):
        defaults = dict(lr=lr, deltaV=deltaV, eps1=eps1, eps2=eps2, deltaEn=deltaEn, weight_decay=weight_decay, eta=eta, consEn=consEn, nu=nu, gamma=gamma, debug=debug)
        self.deltaV = deltaV 
        self.deltaEn = deltaEn
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.debug = debug
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.gamma = gamma
        super(RLSepGen2nd, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        
        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        V = (loss + 0.5*self.weight_decay*self.q2 - self.deltaV)**self.eta

        if self.debug == True:
            print("V: ", V.item())    

        # Initialization and first step
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())

            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(1+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))
                 
            # perform the first half step on the p's
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p.add_(- 0.5 * self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.deltaV))*(d_q+self.weight_decay*q))
    
                    self.p2.add_(torch.norm(p)**2)

            # Update the q's
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    q.data.add_(2 * self.lr * param_state["momenta"]/(1+self.p2) )

            self.iteration += 1

        # The leapfrog loop
        elif V > self.eps2:

            #This is the coefficient multiplying the gradient in the p update rules. 
            # We compute it here only once since it is the same for the two momentum updates
            rhs_coefficient_update_p = - 0.5 * self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.deltaV))

            # Update the p's  and compute p^2 after the integer step and after the total p update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    # Update the p's
                    p = param_state["momenta"]
                    
                    if q.grad is None:
                        continue
                    else:
                        #Integer step
                        p.add_(rhs_coefficient_update_p*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)

            if self.debug :
                print('Relative energy violation:', (torch.log(1+self.p2)+torch.log(V)-self.energy)/self.energy)

            # Check energy conservation after first half of the step and compute scaling factor of the p for energy conservation
            if self.consEn == True:
                    p2true = ((self.expenergy / V)-1)                   
                    if torch.abs(p2true-self.p2) < self.eps1:
                        self.normalization_coefficient = 1.0
                    elif  p2true < 0:
                        self.normalization_coefficient = 1.0
                    else:
                        self.normalization_coefficient = torch.sqrt(p2true / self.p2)
                
            # Update the p's (half step), restore energy conservation,  and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    d_q = q.grad.data 
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(rhs_coefficient_update_p*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)

            # Update the q and slightly rotate the momenta at the same time
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)

            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(1+self.p2))

                    #Add noise to the momentum
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + ((V*(self.deltaEn+1)/self.expenergy)**self.gamma)*self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                
            self.p2 = pnorm**2 
            self.iteration += 1

        return loss

