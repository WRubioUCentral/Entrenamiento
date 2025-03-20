import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from pymongo import MongoClient
import os

# Inicializar el proceso de entrenamiento distribuido
dist.init_process_group("nccl")  # Comunicación optimizada entre GPUs
device = torch.device("cuda")
scaler = GradScaler()  # Para Mixed Precision

# Función de activación personalizada
class FunAct(nn.Module):
    def __init__(self, activacion="sin"):
        super().__init__()
        self.activacion = activacion

    def forward(self, x):
        activaciones = {
            "sin": torch.sin,
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "softmax": lambda x: F.softmax(x, dim=-1),
            "softplus": F.softplus,
        }
        return activaciones.get(self.activacion, torch.sin)(x)

# Conexión a MongoDB
class CargaDB:
    def __init__(self, data):
        try:
            connection_str = os.getenv("MONGO_URI")  # Usar variable de entorno para seguridad
            client = MongoClient(connection_str)
            db = client["data"]
            collection = db["documentos"]
            collection.insert_one(data)
        except Exception as e:
            print("Error de conexión a MongoDB:", e)

# Modelo de entrenamiento
class Modelo(nn.Module):
    def __init__(self, n, funact):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, n), funact,
            nn.Linear(n, n), funact,
            nn.Linear(n, 2)
        )

    def forward(self, x):
        return self.mlp(x)

# Función de entrenamiento
def entrenar(n, funact, N_STEPS, N_SAMPLES, N_SAMPLES_0, u):
    model = Modelo(n, funact).to(device)
    model = DDP(model)  # Habilitar entrenamiento distribuido
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    inicio = time.time()
    for step in range(N_STEPS + 1):
        optimizer.zero_grad()
        
        # Datos de entrenamiento
        X = torch.rand((N_SAMPLES, 2), requires_grad=True).to(device)
        with autocast():
            y_hat = model(X)
            grads, = torch.autograd.grad(y_hat, X, grad_outputs=torch.ones_like(y_hat), create_graph=True)
            dpdx, dpdt = grads[:, 0], grads[:, 1]
            pde_loss = criterion(dpdt, -u * dpdx)
        
        # Condición inicial
        x = torch.rand(N_SAMPLES_0).to(device)
        p0 = torch.sin(2. * np.pi * x).unsqueeze(1)
        X_ini = torch.stack([x, torch.zeros(N_SAMPLES_0).to(device)], axis=-1)
        with autocast():
            ini_loss = criterion(model(X_ini), p0)
        
        # Condiciones de frontera
        t = torch.rand(N_SAMPLES_0).to(device)
        X0 = torch.stack([torch.zeros(N_SAMPLES_0).to(device), t], axis=-1)
        X1 = torch.stack([torch.ones(N_SAMPLES_0).to(device), t], axis=-1)
        with autocast():
            bound_loss = criterion(model(X0), model(X1))
        
        # Backpropagation con Mixed Precision
        loss = pde_loss + ini_loss + bound_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    fin = time.time()
    trainment_time = fin - inicio
    
    # Guardar en MongoDB
    CargaDB({
        'Neuronas': n, 'Funcion_activacion': funact.activacion, 'NSteps': N_STEPS,
        'NSamples': N_SAMPLES, 'NSamples0': N_SAMPLES_0, 'u': u,
        'pde_loss': pde_loss.item(), 'ini_loss': ini_loss.item(),
        'bound_loss': bound_loss.item(), 'loss': loss.item(),
        'trainment_time': trainment_time
    })
    
    print(f'Neurons: {n}, Activation: {funact.activacion}, Loss: {loss.item():.3f}, Time: {trainment_time:.3f}')

# Configuración de entrenamiento en AWS
if __name__ == "__main__":
    for alpha in range(4010, 6010, 10):
        for neuronas in range(100, 550, 50):
            for epocas in range(1000, 6000, 1000):
                for muestras in range(500, 2500, 500):
                    for muestras_0 in range(500, 2500, 500):
                        entrenar(n=neuronas, funact=FunAct("sin"), N_STEPS=epocas, N_SAMPLES=muestras, N_SAMPLES_0=muestras_0, u=alpha)
