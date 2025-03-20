import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from pymongo import MongoClient

class FunAct(torch.nn.Module):                                              ##Módulo función de activación
    def __init__(self, activacion="1"):
        super().__init__()
        self.activacion = str(activacion)

    def forward(self, x):
        activaciones = {
            "sin": torch.sin,
            "tanh": torch.tanh,
            "relu": F.relu, 
            "sigmoid": torch.sigmoid,
            "softplus": F.softplus
        }
        # Obtener la función de activación, por defecto usa torch.sin
        funcion_activacion = activaciones.get(self.activacion, torch.sin)
        return funcion_activacion(x)  # Ejecutar la función aquí
        
class CargaDB:                                                              ##Módulo carga a BD
    """
    n, funact, N_STEPS, N_SAMPLES, N_SAMPLES_0, u, pde_loss, ini_loss, bound_loss, loss, trainment_time, accucary
    """
    def __init__(self, n, funact, N_STEPS, N_SAMPLES, N_SAMPLES_0, u, pde_loss, ini_loss, bound_loss, loss, trainment_time, accuracy):

        try: ## 8OmZTYqNQPeRsLFs
            self.funact = funact
            self.connection_str = "mongodb+srv://williamrubio:8OmZTYqNQPeRsLFs@clusterbigdata.aj1wbar.mongodb.net/?retryWrites=true&w=majority&appName=ClusterBigData"
            self.client = MongoClient(self.connection_str)
            self.db = self.client["data"]
            self.collection = self.db["documentos"]
            #self.client.admin.command("ping")  # Comando para verificar conexión
            #print("\n\n\tConexión a MongoDB exitosa\n\n")

            ##Carga
            data = {'Liberria': 'Torch',
                    'Neuronas': f'{n}',
                    'Funcion_activacion': f'{funact}',
                    'Capas_ocultas': '1',
                    'NSteps' : f'{N_STEPS}',
                    'NSamples': f'{N_SAMPLES}',
                    'NSamples0': f'{N_SAMPLES_0}',
                    'Optimizador': f'Adam',
                    'Funcion_perdida': f'MSE',
                    ##'Momentum': f'0.5',
                    'Learning_rate': f'0.01',
                    'Threshold': f'0.5',
                    'U': f'{u}',
                    'PDE_loss': f'{pde_loss}',
                    'Ini_loss': f'{ini_loss}',
                    'Bound_loss': f'{bound_loss}',
                    'Loss': f'{loss}',
                    'Trainment_time': f'{trainment_time}',
                    'Accuracy': f'{accuracy}'
                    }      
            try:
                self.collection.insert_one(data)
                ## print("Carga exitosa.")
            except Exception as e:
                print("Error de carga a coleccion {self.collection}:", e)

        except Exception as e:
            print("Error de conexion a MongoDB:", e)

class Entrenamiento:

    def calcular_accuracy(self, modelo, X, y_real):                         ##Accuracy
        modelo.eval()
        with torch.no_grad():                                               ##No necesita packpropagation
            y_pred = modelo(X)                                              ##Predice modelo
            error = torch.abs(y_pred - y_real)                              ##Error simple
            threshold = 0.5                                                 ##Nivel de tolerancia
            accuracy = (error < threshold).float().mean().item()
        modelo.train()
        return accuracy
    
    def __init__(self, n, Funact, N_STEPS, N_SAMPLES, N_SAMPLES_0, u):
        self.n = n
        self.Funact = Funact
        self.N_SAMPLES = N_SAMPLES
        self.N_STEPS = N_STEPS
        self.N_SAMPLES_0 = N_SAMPLES_0
        self.u = u

        L, N = 1, 10
        dx, dt = max((L / N) / u, 1e-3), (L / N) / u

        mlp = nn.Sequential(                                                    ## Red Neuronal 1 capa
            nn.Linear(2, self.n), self.Funact,
            nn.Linear(self.n, self.n), self.Funact,
            ##nn.Linear(self.n, self.n), self.Funact,
            nn.Linear(self.n, 1)
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp.to(device)

        inicio = time.time()
        optimizer = torch.optim.Adam(mlp.parameters(), lr = 0.01)##Optimizador
        criterion = torch.nn.MSELoss()                                          ##Función de perdida
        mlp.train()                                                             ##Entrenamiento

        for step in range(N_STEPS + 1):
            ##X = torch.rand((N_SAMPLES, 2), requires_grad=True).to(device)
            X = 2 * (torch.rand((N_SAMPLES, 2), requires_grad=True).to(device) - 0.5) ##Condición inicial normalizada
            y_hat = mlp(X)                                                      ##Modelo
            grads, = torch.autograd.grad(y_hat, X,                              ##Tensores con valores aleatorios en el rango [0, 1)
                                         grad_outputs=torch.ones_like(y_hat),
                                         create_graph=True)
            dpdx, dpdt = grads[:, 0], grads[:, 1]                               ##Derivadas
            pde_loss = criterion(dpdt, -u*dpdx)                               ##Perdidas de la PDE

            x = torch.rand(N_SAMPLES_0).to(device)
            p0 = torch.sin(2. * math.pi * x / L).unsqueeze(1)                   ##Condición inicial
            X_ini = torch.stack([x, torch.zeros(N_SAMPLES_0, device = device)], axis = -1)
            y_hat_ini = mlp(X_ini)                                              ##Modelo con condiciones iniciales
            ini_loss = criterion(y_hat_ini, p0)                                 ##Perdida con condiciones iniciales

            t = torch.rand(N_SAMPLES_0).to(device)
            X0 = torch.stack([torch.zeros(N_SAMPLES_0, device = device), t], axis = -1)
            X1 = torch.stack([torch.ones(N_SAMPLES_0, device = device), t], axis = -1)
            bound_loss = criterion(mlp(X0), mlp(X1))

            optimizer.zero_grad()                                               ##Optimizador
            loss = pde_loss + ini_loss + bound_loss                             ##Perdida total
            loss.backward()                                                     ##Actualiza pesos
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm = 0.5)      ##Evita la explosión de gradientes
            optimizer.step()                                                    ##Optimizador de épocas

            accuracy = self.calcular_accuracy(mlp, X_ini, p0)                   ##Accuracy

            fin = time.time()
            trainment_time = fin - inicio
            #if (step != N_STEPS):
            ## print(f'Alpha {u} Neuronas {n} Funact {Funact.activacion} Paso {step} Epocas {N_STEPS} Ini_oss {ini_loss.item():.4f} Bound_loss {bound_loss.item():.4f} Pde_oss {pde_loss.item():.4f} Loss {loss.item():.4f} Entrenamiento {trainment_time:.3f} Accuracy: {accuracy:.3%}')

        CargaDB(
            n=n, funact=Funact.activacion, N_STEPS=N_STEPS, N_SAMPLES=N_SAMPLES,
            N_SAMPLES_0=N_SAMPLES_0, u=u, pde_loss=f'{pde_loss:4f}', ini_loss=f'{ini_loss:4f}',
            bound_loss=f'{bound_loss:4f}', loss=f'{loss:4f}', trainment_time=f'{trainment_time:4f}', accuracy=f'{accuracy:4f}'
        )
        print(f'Alpha {u} Neuronas {n} Funact {Funact.activacion} Epocas {N_STEPS} Loss {loss.item():.4f} Entrenamiento {trainment_time:.4f} Accuracy: {accuracy:.4%}')


for alpha in range(300, 5100, 100):
    for funcion_activacion in ["tan", "sin", "relu", "sigmoid", "softplus"]:
        for neuronas in range(200, 5500, 500):
            for epocas in range(200, 10400, 200):
                Inicio = Entrenamiento(n = neuronas,
                                        Funact = FunAct(activacion = funcion_activacion),
                                        N_STEPS = epocas,
                                        N_SAMPLES = 1000,
                                        N_SAMPLES_0 = 1000,
                                        u = alpha)
