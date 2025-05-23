import torch
from collections import namedtuple
import ipdb
from typing import List, Callable
import time
from torch.func import hessian, vmap, jacrev

QuadCost = namedtuple("QuadCost", "C c")
LinDx = namedtuple("LinDx", "F f")

# https://stackoverflow.com/questions/11351032
QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)


def merit_function(
    xu, Q, q, dx, x0, lamda, rho, x_lower, x_upper, diag_cost=True
):

    if xu.dim() == 4:
        n_outs, bsz = xu.shape[:2]
        x0 = x0[None].repeat(n_outs, 1, 1).view(n_outs * bsz, x0.shape[1])
        xu = xu.view(n_outs * bsz, xu.shape[2], xu.shape[3])
        Q = Q[None].repeat(n_outs, 1, 1, 1).view(
            n_outs * bsz, Q.shape[1], Q.shape[2])
        q = q[None].repeat(n_outs, 1, 1, 1).view(
            n_outs * bsz, q.shape[1], q.shape[2])
        rho = rho[None].repeat(n_outs, 1, 1).view(n_outs * bsz, rho.shape[1])
        lamda = lamda[None].repeat(n_outs, 1, 1).view(
            n_outs * bsz, lamda.shape[1])
        bsz = n_outs * bsz
    else:
        bsz = xu.size(0)
    x = xu[:, :, : x0.size(-1)]
    cost_total = compute_cost(x, Q, q)
    res, res_clamp = dyn_res(xu, dx, x0, x_lower, x_upper)
    merit_value = cost_total + 0.5 * rho[:, 0] * (res_clamp * res_clamp).view(
        bsz, -1).sum(dim=1) + (lamda * res).view(bsz, -1).sum(dim=1)
    return merit_value



def merit_grad_hessian(
    xu,
    Q,
    q,
    dx,
    dx_jac,
    x0,
    lamda,
    rho,
    x_lower,
    x_upper,
    diag_cost=True,
):
    bsz = xu.size(0)
    xsize = x0.size(-1)
    x, u = xu[:, :, :xsize], xu[:, :, xsize:]
    res, res_clamp, constraint_jac, constraint_jac_clamp, constraint_hess = (
        constraint_res_jac2(xu, x0, dx_jac, x_lower, x_upper)
    )

    merit_grad = (
        compute_cost_gradient(x, Q, q, diag_cost).view(bsz, -1)
        + (lamda[..., None] * constraint_jac).sum(dim=-2)
        + rho * (res_clamp[..., None] * constraint_jac_clamp).sum(dim=-2)
    )
    if diag_cost:
        Qfull = torch.diag_embed(Q.reshape(bsz, -1))
    merit_hess = Qfull + rho[:, :, None] * constraint_hess
    merit_hess_clip = Qfull + torch.clamp(rho[:, :, None], max=10) * constraint_hess
    # print(constraint_hess.norm().item(), Qfull.norm().item(), constraint_jac_clamp.norm().item(), constraint_jac.norm().item())
    return merit_grad, merit_hess, merit_hess#_clip


def merit_hessian(
    xu, Q, q, dx_jac, x0, lamda, rho, x_lower, x_upper, u_lower, u_upper, diag_cost=True
):
    bsz = xu.size(0)
    xsize = x0.size(-1)
    res, res_clamp, constraint_jac, constraint_jac_clamp = constraint_res_jac2(
        xu, x0, dx_jac, x_lower, x_upper, u_lower, u_upper
    )
    constraint_hess = torch.bmm(
        constraint_jac_clamp.permute(0, 2, 1), constraint_jac_clamp
    )
    if diag_cost:
        Qfull = torch.diag_embed(Q.reshape(bsz, -1))
    merit_hess = Qfull + rho[:, :, None] * constraint_hess
    return merit_hess

def constraint_res_jac2(xu, x0, dx_jac, x_lower, x_upper):
    x_size = x0.size(-1)
    x, u = xu[:, :, :x_size], xu[:, :, x_size:]

    res_eq, res_eq_jac = dyn_res_eq_jac(x, u, dx_jac, x0)
    # res_ineq, res_ineq_clamp, res_ineq_jac, res_ineq_jac_clamp = dyn_res_ineq_jac(
    #     x, u, x0, x_lower, x_upper
    # )
    # return constraint_res_jac2_jit(
    #     res_eq, res_ineq, res_eq_jac, res_ineq_jac, res_ineq_clamp, res_ineq_jac_clamp
    # )

    constraint_hess = torch.bmm(res_eq_jac.permute(0, 2, 1), res_eq_jac)
    return res_eq, res_eq, res_eq_jac, res_eq_jac, constraint_hess

def constraint_res_jac2_jit(
    res_eq, res_ineq, res_eq_jac, res_ineq_jac, res_ineq_clamp, res_ineq_jac_clamp
):
    res = torch.cat((res_eq, res_ineq), dim=-1)
    res_clamp = torch.cat((res_eq, res_ineq_clamp), dim=-1)
    constraint_jac = torch.cat((res_eq_jac, res_ineq_jac), dim=1)
    constraint_jac_clamp = torch.cat((res_eq_jac, res_ineq_jac_clamp), dim=1)
    constraint_hess = torch.bmm(
        constraint_jac_clamp.permute(0, 2, 1), constraint_jac_clamp
    )
    return res, res_clamp, constraint_jac, constraint_jac_clamp, constraint_hess


def dyn_res_eq(
    x: torch.Tensor, u: torch.Tensor, dx: torch.nn.Module, x0: torch.Tensor
) -> torch.Tensor:
    "split x into state and control and compute dynamics residual using dx"
    bsz, T, xsize = x.size()
    _, _, usize = u.size()
    x_next = dx(x[:, :-1].reshape(-1, xsize), u[:, :-1].reshape(-1, usize)).view(
        bsz, T - 1, xsize
    )
    res_d = x[:, 1:, :] - x_next
    # if self.add_goal_constraint:
    #     res_goal = (x[:,-1,:] - self.x_goal).reshape(bsz, -1)
    #     res = torch.cat((res, res_init, res_goal), dim=1)
    # else:
    # res = torch.cat((res_d, res_init), dim=1)
    res = res_d.reshape(bsz, -1)
    return res


def block_diag(mats):
    return torch.block_diag(*mats)


def dyn_res_eq_jac(x, u, dx_jac, x0):
    bsz, T, x_size = x.shape
    u_size = u.shape[-1]
    x_next, dynamics_jacobian = dx_jac(
        x[:, :-1].reshape(-1, x_size), u[:, :-1].reshape(-1, u_size)
    )
    # x_next = x[:,1:]
    # dynamics_jacobian = dx_jac
    dynamics_jacobian = torch.cat(
        (
            dynamics_jacobian[0].reshape(bsz, T - 1, x_size, x_size),
            dynamics_jacobian[1].reshape(bsz, T - 1, x_size, u_size)*0.0,
        ),
        dim=-1,
    )
    dynamics_jacobian = torch.vmap(block_diag)(dynamics_jacobian)
    id_x = [
        torch.cat([torch.eye(x_size), torch.zeros(
            (x_size, u_size))], dim=1).to(x)
    ] * (T - 1)
    id_x = torch.block_diag(*id_x)
    return dyn_res_eq_jac_mat_filling(
        x,
        dynamics_jacobian,
        x0,
        x_next.view(bsz, T - 1, x_size),
        id_x,
        x_size,
        u_size,
        bsz,
        T,
    )


# @torch.jit.script
def dyn_res_eq_jac_mat_filling(
    x, dynamics_jacobian, x0, x_next, id_x, x_size: int, u_size: int, bsz: int, T: int
):
    x_res = x[:, 1:, :] - x_next
    res = x_res.reshape(bsz, -1)
    constraint_jacobian = torch.zeros(
        bsz, (T-1) * x_size, T * (x_size + u_size)).to(x)
    id_x = id_x[None].repeat(bsz, 1, 1)
    constraint_jacobian[:, :, x_size + u_size:] = id_x
    constraint_jacobian[:, :, : -
                        (x_size + u_size)] += -dynamics_jacobian
    return res, constraint_jacobian


# @torch.jit.script
def dyn_res_ineq(x, u, x0, x_lower, x_upper, u_lower, u_upper):
    bsz = x.size(0)
    # res = None
    # Add control constraints if self.u_lower and self.u_upper are not None
    # if u_lower is not None:
    res = torch.cat((u - u_upper, -u + u_lower), dim=2)
    # if x_lower is not None:
    #     res_x = torch.cat((
    #         x - x_upper,
    #         -x + x_lower
    #         ), dim=2)
    #     if res is None:
    #         res = res_x
    #     else:
    #         res = torch.cat((res,res_x), dim=2)

    # Add other inequality constraints if self.ineqG and self.ineqh are not None
    # if self.ineqG is not None:
    #     res_G = torch.bmm(self.ineqG, x.unsqueeze(-1)) - self.ineqh
    #     if res is None:
    #         res = res_G
    #     else:
    #         res = torch.cat((res,res_G), dim=2)
    res = res.reshape(bsz, -1)
    res_clamp = torch.clamp(res, min=0)
    return res, res_clamp


def dyn_res_ineq_jac(x, u, x0, x_lower, x_upper, u_lower, u_upper):
    bsz, T, x_size = x.shape
    res, res_clamp, id_xu, clamp_idx = dyn_res_ineq_jac_jit(
        x, u, x0, x_lower, x_upper, u_lower, u_upper
    )
    id_xu = torch.block_diag(*id_xu)
    id_xu = id_xu[None].repeat(bsz, 1, 1)
    # ipdb.set_trace()
    # id_xu_clamp = id_xu * (res_clamp > 0).float()[..., None]
    id_xu_clamp = id_xu * clamp_idx.float()[..., None]
    # print("ineqjac :", id_xu_clamp.norm().item(), id_xu.norm().item(), res_clamp.norm().item(), res.norm().item())
    # ipdb.set_trace()
    return res, res_clamp, id_xu, id_xu_clamp


@torch.jit.script
def dyn_res_ineq_jac_jit(x, u, x0, x_lower, x_upper, u_lower, u_upper):
    bsz, T, x_size = x.shape
    u_size = u.shape[-1]
    res = None
    res = torch.cat((u - u_upper, -u + u_lower), dim=2)
    res = res.reshape(bsz, -1)
    clamp_idx = res >= 0
    res_clamp = torch.clamp(res, min=0)
    id_u = torch.eye(u_size, device=x.device).to(x)
    id_u = torch.cat((id_u, -id_u), dim=0)
    id_x = torch.zeros((2 * u_size, x_size), device=x.device).to(x)
    id_xu = torch.cat((id_x, id_u), dim=1)
    id_xu = [id_xu] * T
    return res, res_clamp, id_xu, clamp_idx


def dyn_res(xu, dx, x0, x_lower=None, x_upper=None):
    x_size = x0.size(-1)
    bsz, T, xu_size = xu.shape
    xu = xu.reshape(-1, T, xu_size)
    x, u = xu[:, :, :x_size], xu[:, :, x_size:]
    # Equality residuals
    res_eq = dyn_res_eq(x, u, dx, x0)
    # Inequality residuals
    # ipdb.set_trace()

    return res_eq, res_eq
    # res_ineq, res_ineq_clamp = dyn_res_ineq(
    #     x, u, x0, x_lower, x_upper, u_lower, u_upper
    # )
    # return torch.cat((res_eq, res_ineq), dim=1), torch.cat(
    #     (res_eq, res_ineq_clamp), dim=1
    # )


@torch.jit.script
def compute_cost(
    x: torch.Tensor, Q: torch.Tensor, q: torch.Tensor, diag_cost: bool = True
) -> torch.Tensor:
    C = Q
    c = q
    xsize = x.size(-1)
    # ipdb.set_trace()
    if diag_cost:
        C = C[:, :, :xsize]
        c = c[:, :, :xsize]
        return (0.5 * (x * C * x).sum(-1) + (c * x).sum(-1)).sum(dim=-1)
    C = C[:, :, :xsize, :xsize]
    c = c[:, :, :xsize]
    return 0.5 * ((x.unsqueeze(-1) * C).sum(dim=-2) * x).sum(dim=-1).sum(dim=-1) + (
        x * c
    ).sum(dim=-1).sum(dim=-1)


@torch.jit.script
def compute_cost_gradient(
    x: torch.Tensor, Q: torch.Tensor, q: torch.Tensor, diag_cost: bool = True
) -> torch.Tensor:
    C = Q.clone()
    c = q.clone()
    xsize = x.size(-1)
    usize = C.size(-1) - xsize
    xu = torch.cat((x, torch.zeros_like(c[:, :, :usize])), dim=-1)
    if diag_cost:
        C[:, :, xsize:] *= 0
        c[:, :, xsize:] *= 0
        return C * xu + c
    C[:, :, xsize:, xsize:] *= 0
    c[:, :, xsize:] *= 0
    return torch.cat((C * xu, c), dim=-1)


class NewtonAL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        meritfn,
        dyn_fn,
        cost_fn,
        merit_grad_hessfn,
        xi,
        x0,
        lam,
        rho,
        Q,
        q,
        threshold,
        eps,
        ls,
        verbose
    ):
        bsz, T, n_elem = xi.size()  # (bsz, T, xd+ud)
        dev = xi.device

        def meritGHfnQ(x): return merit_grad_hessfn(x, Q, q, lam)
        def cost_fnQ(x): return cost_fn(x, Q, q)
        def meritfnQ(x): return meritfn(x, Q, q, lam, x0, rho)
        # meritfn_mean = lambda x, Qi, qi, yi, x0i, rhoi: meritfn(x.view((1,T,n_elem)), Qi[None].view((1,T,n_elem)), qi[None].view((1,T,n_elem)), yi[None], x0i[None], rhoi[None], grad=True).mean()

        x_est = xi  # (bsz, 2d, L')
        cost = cost_fnQ(x_est)
        dyn_res = dyn_fn(x_est)
        # ipdb.set_trace()
        merit = meritfn(x_est, Q, q, lam, x0, rho)

        # Solve for newton steps on the augmented lagrangian
        nstep = 0
        max_newton_steps = 4  # maximum number of Newton steps for each AL step
        old_dyn_res = torch.norm(dyn_res).item()
        if verbose:
            print(nstep, (dyn_res.view(bsz, -1).norm(dim=-1)).mean().item(), (cost).mean().item(), merit.mean().item())
        stepsz = 1
        cholesky_fail = torch.tensor(True)
        merit_delta = 1
        # while (
        #         merit_delta > threshold*1e-8 and nstep < max_newton_steps):  # and stepsz > 1e-8
            # ):  # and update_norm > 1e-3*init_update_norm:
        while nstep < max_newton_steps and stepsz > 1e-8:
            # ipdb.set_trace()
            nstep += 1
            # Compute the hessian and gradient of the augmented lagrangian
            with torch.enable_grad():
                x_est.requires_grad_(True)
                grad, Hess, Hess_clip = meritGHfnQ(x_est)
            # ipdb.set_trace()
            # Solve for the newton step
            stepsz = 0
            # Hess_clip = Hess = torch.eye(Hess.shape[-1], device=Hess.device, dtype=Hess.dtype).repeat(bsz, 1, 1) 
            if not cholesky_fail:
                U, info = torch.linalg.cholesky_ex(Hess)
                update = -torch.cholesky_solve(grad.reshape(bsz, -1, 1), U).reshape(
                    bsz, T, n_elem
                )
                if update.isnan().sum() > 0 or update.isinf().sum() > 0:
                    update = -torch.linalg.solve(Hess, grad.reshape(bsz, -1)).reshape(
                        bsz, T, n_elem
                    )
                    cholesky_fail = torch.tensor(True)
            else:
                U = Hess
                # update = -grad.reshape(#torch.linalg.solve(Hess, grad.reshape(bsz, -1)).reshape(
                #     bsz, T, n_elem
                # )
                update = -torch.linalg.solve(Hess, grad.reshape(bsz, -1)).reshape(
                    bsz, T, n_elem
                )
                # update = torch.round(update, decimals=2)

            if ls:
                x_est1, new_merit, stepsz, status = line_search_newton(
                    update, x_est, meritfnQ, merit, x0
                )
            else:
                x_est1 = x_est + update
                new_merit = meritfnQ(x_est)

            if (
                x_est1.isnan().sum() > 0 or x_est1.isinf().sum() > 0
            ):  # or new_merit.isnan().sum() > 0 or new_merit.isinf().sum() > 0:
                ipdb.set_trace()
            cost = cost_fnQ(x_est1)
            dyn_res = dyn_fn(x_est1)
            new_dyn_res = torch.norm(dyn_res).item()
            if verbose:
                print(nstep, (dyn_res.view(bsz, -1).norm(dim=-1)).mean().item(), (cost).mean().item(), torch.norm(update).item(), new_merit.mean().item(), stepsz)

            ## exit creteria
            x_est = x_est1
            if (
                abs(old_dyn_res - new_dyn_res) / new_dyn_res < 1e-3
                or new_dyn_res < 1e-3
            ):
                break

            old_dyn_res = new_dyn_res
            # 
            merit_delta = 1000#((new_merit - merit) / new_merit).abs().max().item()#1000
            merit = new_merit

        try:
            ctx.save_for_backward(Hess_clip, U, x_est, cholesky_fail)
        except:
            ipdb.set_trace()
        Us, VTs = None, None
        return x_est, status

    @staticmethod
    def backward(ctx, x_grad, status_grad):
        # implicit gradients w.r.t Q and q
        H, U, x, cholesky_fail = ctx.saved_tensors
        bsz = x_grad.size(0)

        # solve Hx + g = 0, H = d^2f/dx^2, g is x_grad
        if cholesky_fail:
            inp_grad = -torch.linalg.solve(H, x_grad.view(bsz, -1)).reshape(
                x_grad.shape
            )
        else:
            inp_grad = -torch.cholesky_solve(x_grad.view(bsz, -1, 1), U).reshape(
                x_grad.shape
            )

        # Compute the gradient w.r.t. the Q and q
        Q_grad = inp_grad * x  # if Q is diag
        # Q_grad = torch.bmm(inp_grad, x.transpose(1,2)) # if Q is not diag
        q_grad = inp_grad

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Q_grad,
            q_grad,
            None,
            None,
            None,
            None
        )


def line_search_newton(update, x_est, meritfnQ, merit, x0):
    n_ls = 20  # TODO: make this a parameter
    xsize = x0.shape[-1]
    stepsz = torch.ones(x_est.shape[0], device=x_est.device) * 2
    mask = torch.ones(x_est.shape[0], device=x_est.device)
    stepszs = 2 ** (
        -torch.arange(n_ls, device=x_est.device)
        .float()
        .unsqueeze(1)
        .expand(n_ls, x_est.shape[0])
    )
    x_next = x_est[None] + stepszs[:, :, None, None] * update[None]
    # x_next[:, :, 0, :xsize] = x0[None]
    # new2_objective = torch.stack([meritfnQ(x_next[i]) for i in range(n_ls)], dim=0)
    # new2_objective = torch.vmap(meritfnQ)(x_next)
    new2_objective = meritfnQ(x_next).reshape(n_ls, -1)
    new2_objective_min = torch.min(new2_objective, dim=0)
    batch_idxs = torch.arange(x_est.shape[0], device=x_est.device)
    stepsz = stepszs[new2_objective_min.indices, batch_idxs]
    x_next = x_next[new2_objective_min.indices, batch_idxs]
    new2_objective = new2_objective_min.values
    status = (new2_objective < merit).float()
    x_est = status[:, None, None] * x_next + \
        (1 - status)[:, None, None] * x_est
    return x_est, new2_objective, stepsz.mean().item(), status
