import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

def model_pred(model, Lx, Ly, N=100, bar_max=0, title='Point Source Helmholtz', filename='Helmholtz'):
    x, y = torch.linspace(-Lx, Lx, N), torch.linspace(-Ly, Ly, N)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    model = model.to('cpu')
    inputs = torch.cat([x, y], dim=-1)
    p_real = model(inputs)[:, 0].detach().numpy()
    p_imag = model(inputs)[:, 1].detach().numpy()
    p_real, p_imag = p_real.reshape(N, N), p_imag.reshape(N, N)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    if bar_max == 0:
        p_max = np.max(np.abs(p_real))
    else:
        p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), p_real, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title(f"{title} (real)")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    if bar_max == 0:
        p_max = np.max(np.abs(p_imag))
    else:
        p_max = bar_max
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x.reshape(N, N), y.reshape(N, N), p_imag, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title(f"{title} (imag)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"{filename}.jpg")
    plt.tight_layout()
    
def plot_field(field, Lx, Ly, bar_max=0, title='Point Source Helmholtz', filename='reference'):
    Nx, Ny = field.shape[0], field.shape[1]   
    x, y = np.linspace(-Lx, Lx, Nx), np.linspace(-Ly, Ly, Ny)
    x, y = np.meshgrid(x, y)
    p_real, p_imag = np.real(field), np.imag(field)
    
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
    plt.xlabel('x')
    plt.ylabel('y')

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
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f"{filename}.jpg")
    
def plot_error_field(p_pred, p_ref, Lx, Ly):
    Nx, Ny = p_ref.shape[0], p_ref.shape[1]   
    x, y = np.linspace(-Lx, Lx, Nx), np.linspace(-Ly, Ly, Ny)
    x, y = np.meshgrid(x, y)
    p = p_pred - p_ref
    p_real, p_imag = np.real(p), np.imag(p)
    
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    p_max = np.max(np.abs(p_real))
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x, y, p_real, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('Error (real)')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    p_max = np.max(np.abs(p_imag))
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    plt.contourf(x, y, p_imag, levels=400, cmap=cmap, origin='lower', norm=norm)
    plt.colorbar()
    plt.title('Error (imag)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('error.jpg')
    plt.tight_layout()
    
def plot_loss(l_lst):
    l = np.zeros(len(l_lst))
    i = 0
    for z in l_lst:
        l[i] = z
        i = i + 1
    plt.plot(np.arange(0, len(l)), l)
    plt.savefig('loss')
