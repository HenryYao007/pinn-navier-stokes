"""
Physics-informed neural network for the 2D incompressible Navier-Stokes equations
(cylinder wake, continuous-time identification), implemented in PyTorch.

Adapted from the original TensorFlow 1 implementation by Maziar Raissi, Paris Perdikaris,
and George Em Karniadakis: https://github.com/maziarraissi/PINNs (MIT License)

PyTorch adaptation: Henry Yao
Preserves: psi/p outputs, tanh hidden layers with a linear output, PDE residual loss,
and the Adam -> L-BFGS training schedule.
"""

import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.autograd import grad

torch.manual_seed(1234)
np.random.seed(1234)

def xavier_init_(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

class MLP(nn.Module):
    #Fully-connected tanh network with linear output (no activation on last layer)
    def __init__(self, layers):
        super().__init__()
        L = []
        for i in range(len(layers) - 2):
            L += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        L += [nn.Linear(layers[-2], layers[-1])]  # linear output: [psi, p]
        self.net = nn.Sequential(*L)
        self.apply(xavier_init_)

    def forward(self, x):
        return self.net(x)  # returns [psi, p]

class PhysicsInformedNN(nn.Module):
    """
    Inputs: x,y,t -> outputs: u,v,p via stream function psi and pressure p
    Loss = MSE(u_data - u_pred) + MSE(v_data - v_pred) + MSE(f_u) + MSE(f_v)
    with f_u, f_v Navier–Stokes residuals.
    """
    def __init__(self, X, u, v, layers, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        # bounds for input normalization to [-1, 1]
        self.lb = torch.tensor(X.min(0), dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32, device=self.device)

        self.register_buffer("lb_buf", self.lb)
        self.register_buffer("ub_buf", self.ub)

        self.net = MLP(layers).to(self.device)

        # learnable PDE parameters (lambda_1 = Reynolds-like convective coeff, lambda_2 = viscosity)
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device))

        # stash training data (requires_grad True for x,y,t to take derivs)
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=self.device, requires_grad=True)
        self.y = torch.tensor(X[:, 1:2], dtype=torch.float32, device=self.device, requires_grad=True)
        self.t = torch.tensor(X[:, 2:3], dtype=torch.float32, device=self.device, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, device=self.device)
        self.v = torch.tensor(v, dtype=torch.float32, device=self.device)

    def _normalize(self, X):
        # map inputs to [-1, 1] using train-set bounds
        return 2.0 * (X - self.lb_buf) / (self.ub_buf - self.lb_buf) - 1.0

    def forward(self, x, y, t):
        """Return u, v, p plus residuals f_u, f_v for given (x,y,t) (all tensors require_grad)."""
        X = torch.cat([x, y, t], dim=1)
        Xn = self._normalize(X)
        psi_p = self.net(Xn)
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]

        # stream-function -> velocities
        u = grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x, torch.ones_like(psi), create_graph=True)[0]

        # first derivatives
        u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, torch.ones_like(u), create_graph=True)[0]

        v_t = grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, torch.ones_like(v), create_graph=True)[0]

        # second derivatives
        u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

        # pressure gradients
        p_x = grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, torch.ones_like(p), create_graph=True)[0]

        # Navier–Stokes residuals
        l1 = self.lambda_1
        l2 = self.lambda_2
        f_u = u_t + l1 * (u * u_x + v * u_y) + p_x - l2 * (u_xx + u_yy)
        f_v = v_t + l1 * (u * v_x + v * v_y) + p_y - l2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss(self):
        u_pred, v_pred, _, f_u, f_v = self.forward(self.x, self.y, self.t)
        mse = nn.MSELoss()
        data_loss = mse(u_pred, self.u) + mse(v_pred, self.v)
        pde_loss = torch.mean(f_u**2) + torch.mean(f_v**2)
        return data_loss + pde_loss, data_loss.detach(), pde_loss.detach()

    # @torch.no_grad(). In PINNs, we need grad
    def predict(self, x_star, y_star, t_star):
        x = torch.tensor(x_star, dtype=torch.float32, device=self.device, requires_grad=True)
        y = torch.tensor(y_star, dtype=torch.float32, device=self.device, requires_grad=True)
        t = torch.tensor(t_star, dtype=torch.float32, device=self.device, requires_grad=True)
        u, v, p, _, _ = self.forward(x, y, t)
        return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()

def train(model, adam_iters=20000, lbfgs_iters=50000, print_every=1000):
    # Adam warm-up
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for it in range(adam_iters):
        opt.zero_grad()
        L, Ld, Lf = model.loss()
        L.backward()
        opt.step()
        if (it % print_every) == 0:
            l1 = model.lambda_1.item()
            l2 = model.lambda_2.item()
            print(f"[Adam] it={it:6d}  loss={L.item():.3e}  data={Ld:.3e}  pde={Lf:.3e}  l1={l1:.3f}  l2={l2:.5f}  dt={time.time()-t0:.1f}s")
            t0 = time.time()

    # L-BFGS polish (PyTorch needs a closure)
    def closure():
        opt_lbfgs.zero_grad()
        L, _, _ = model.loss()
        L.backward()
        return L

    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_iters,
        max_eval=lbfgs_iters,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        lr=1.0
    )
    L = opt_lbfgs.step(closure)
    with torch.no_grad():
        l1 = model.lambda_1.item()
        l2 = model.lambda_2.item()
    print(f"[LBFGS] final_loss={float(L):.3e}  l1={l1:.3f}  l2={l2:.5f}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # network architecture
    layers = [3, 36, 36, 36, 36, 36, 36, 2]

    # Load data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T (not used in training here; optional supervision)
    t_star = data['t']       # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Tile onto space-time grid
    XX = np.tile(X_star[:, 0:1], (1, T))     # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))     # N x T
    TT = np.tile(t_star, (1, N)).T           # N x T
    UU = U_star[:, 0, :]                     # N x T
    VV = U_star[:, 1, :]                     # N x T

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]

    # sample training points
    N_train = 5000
    idx = np.random.choice(N * T, N_train, replace=False)
    X_train = np.concatenate([x[idx, :], y[idx, :], t[idx, :]], axis=1)
    u_train = u[idx, :]
    v_train = v[idx, :]

    model = PhysicsInformedNN(X_train, u_train, v_train, layers, device=device)
    train(model, adam_iters=4000, lbfgs_iters=1000, print_every=10)

    # Prediction at one snapshot time
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]


    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    error_lambda_1 = np.abs(model.lambda_1.item() - 1.0) * 100
    error_lambda_2 = np.abs(model.lambda_2.item() - 0.01) / 0.01 * 100

    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))