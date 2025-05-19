import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from utils.functions import green

def model_pred(model, Lx, Ly, L_pml, guide_mode='half', bar_max=0.2, scale=100, N=100, is_inverse=True):
    if guide_mode == 'half':
        x, y = torch.linspace(-(Lx - L_pml), (Lx - L_pml), N), torch.linspace(-(Ly - L_pml), Ly, N)
    if guide_mode == 'ideal':
        x, y = torch.linspace(-(Lx - L_pml), (Lx - L_pml), N), torch.linspace(-Ly, Ly, N)
    if guide_mode == 'pekeris':
        x, y = torch.linspace(-(Lx - L_pml), (Lx - L_pml), N), torch.linspace(-(Ly - L_pml), Ly, N)
    if guide_mode == 'free':
        x, y = torch.linspace(-Lx, Lx, N), torch.linspace(-Ly, Ly, N)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    model = model.to('cpu')
    inputs = torch.cat([x, y], dim=-1)
    p_real = model(inputs)[:, 0].detach().numpy()
    if is_inverse:
        p_real = -p_real
    p_imag = model(inputs)[:, 1].detach().numpy()
    p_real, p_imag = p_real.reshape(N, N), p_imag.reshape(N, N)

    if guide_mode != 'free':
        x, y = (x+1) * scale, -(y-1) * scale
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), p_real, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('PINN (real)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), p_imag, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('PINN (imag)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('PINN.jpg')
    
def model_pred_green(model, Lx, Ly, L_pml, strentch, src, k, guide_mode='half', bar_max=0.2, scale=100, N=100, is_inverse=True):
    if guide_mode == 'half':
        x, y = torch.linspace(-(Lx - L_pml), (Lx - L_pml), N), torch.linspace(-(Ly - L_pml), Ly, N)
    if guide_mode == 'ideal':
        x, y = torch.linspace(-(Lx - L_pml), (Lx - L_pml), N), torch.linspace(-Ly, Ly, N)
    if guide_mode == 'pekeris':
        x, y = torch.linspace(-(Lx - L_pml), (Lx - L_pml), N), torch.linspace(-(Ly - L_pml), Ly, N)
    if guide_mode == 'free':
        x, y = torch.linspace(-Lx, Lx, N), torch.linspace(-Ly, Ly, N)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    model = model.to('cpu')
    inputs = torch.cat([x, y], dim=-1)
    p_real = model(inputs)[:, 0].detach().numpy()
    if is_inverse:
        p_real = -p_real
    p_imag = model(inputs)[:, 1].detach().numpy()
    g = green(x, y, src, k, strentch=strentch, device='cpu').numpy()
    p_real, p_imag = p_real.reshape(N, N) + g[:, 0].reshape(N, N), p_imag.reshape(N, N) + g[:, 1].reshape(N, N)

    if guide_mode != 'free':
        x, y = (x+1) * scale, -(y-1) * scale
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), p_real, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('PINN (real)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), p_imag, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('PINN (imag)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('PINN.jpg')
    
def plot_field(field, Lx, Ly, bar_max=0, title='Point Source Helmholtz', filename='reference'):
    Nx, Ny = field.shape[0], field.shape[1]   
    x, y = np.linspace(Lx[0], Lx[1], Nx), np.linspace(Ly[0], Ly[1], Ny)
    x, y = np.meshgrid(x, y)
    p_real, p_imag = np.real(field), np.imag(field)
    y = -y
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    if bar_max == 0:
        p_max = np.max(np.abs(p_real))
    else:
        p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x, y, p_real, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title(f"{title} (real)")
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    if bar_max == 0:
        p_max = np.max(np.abs(p_imag))
    else:
        p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x, y, p_imag, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title(f"{title} (imag)")
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{filename}.jpg")
    
def plot_error(p_pred, p_ref, x, y, guide_mode='half', bar_max=0.2, scale=100, N=100, is_inverse='True'):
    p_pred_real, p_pred_imag = np.real(p_pred), np.imag(p_pred)
    p_ref_real, p_ref_imag = np.real(p_ref), np.imag(p_ref)
    error_real = p_pred_real - p_ref_real
    error_imag = p_pred_imag - p_ref_imag

    if guide_mode != 'free':
        x, y = (x+1) * scale, -(y-1) * scale
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), error_real, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('Error (real)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), error_imag, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('Error (imag)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Error.jpg')
    print(f"Error(Real) {np.abs(p_pred_real - p_ref_real).mean() / np.abs(p_ref_real).mean()}, Error(Imag) {np.abs(p_pred_imag - p_ref_imag).mean() / np.abs(p_ref_imag).mean()}")
    
def plot_loss(l_lst):
    l = np.zeros(len(l_lst))
    i = 0
    for z in l_lst:
        l[i] = z
        i = i + 1
    plt.plot(np.arange(0, len(l)), l)
    plt.savefig('loss')
