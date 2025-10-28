# 221840194 Runbang Wang
# cal1_Bubble_Vibration.py
import numpy as np
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self,
                 R0: float = 1.5e-6, # 平衡半径
                 omega: float = 5e6, 
                 rho_L: float = 1000.0, # 流体密度 (kg/m^3)
                 sigma: float = 0.07275, # 表面张力 (N/m)
                 PV: float = 2330, #蒸汽压
                 P0: float = 1.013e5,
                 Pa: float = 1e-7, # 外部声压
                 gamma:float = 1.4, # 气体比热容
                 # Pr: float = 1 # 课堂上给出的公式中有此项，但是实在找不到到底是哪一个量，所以暂时不考虑
                 ):       
        self.R0 = R0
        self.omega = omega
        self.rho_L = rho_L
        self.sigma = sigma
        self.PV = PV
        self.P0 = P0
        self.Pa = Pa
        self.gamma = gamma
        self.Pg = P0 - PV + 2*sigma/R0

def Derivative_Function(p: Parameters, x: np.ndarray, t: float):
    # x contains R and U
    # Dx contains dR_dt and dU_dt
    R = x[0]
    U = x[1]
    
    dR_dt = U
    dU_dt = ( -3/2*p.rho_L*U**2 + p.Pg*(p.R0/R)**(3*p.gamma) - p.P0 - 2*p.sigma/R 
            - p.sigma*p.omega*p.rho_L*R*U - p.Pa*np.cos(p.omega*t) ) / (p.rho_L*R)

    Dx = np.array([dR_dt, dU_dt])
    return Dx


class Integrator:
    def __init__(self, p: Parameters, x0: np.ndarray, 
                t_start: float, t_end: float, dt: float, 
                Intergrator_Method: str = "RK4"):    
        self.p = p
        self.x0 = x0
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.Intergrator_Method = Intergrator_Method

    def integrator(self):
        t_steps = int ((self.t_end - self.t_start) / self.dt)
        t_array = np.linspace(self.t_start, self.t_end, t_steps)
        x_array = np.zeros((t_steps, self.x0.size))
        x_array[0, :] = self.x0

        for i in range(1, t_steps):
            if self.Intergrator_Method == "RK1":
                x_array[i, :] = self.RK1(self.p, x_array[i-1, :], t_array[i-1], self.dt)
            elif self.Intergrator_Method == "RK2":
                x_array[i, :] = self.RK2(self.p, x_array[i-1, :], t_array[i-1], self.dt)
            elif self.Intergrator_Method == "RK3":
                x_array[i, :] = self.RK3(self.p, x_array[i-1, :], t_array[i-1], self.dt)
            elif self.Intergrator_Method == "RK4":
                x_array[i, :] = self.RK4(self.p, x_array[i-1, :], t_array[i-1], self.dt)
            elif self.Intergrator_Method == "RK4_2":
                x_array[i, :] = self.RK4_2(self.p, x_array[i-1, :], t_array[i-1], self.dt)
            else:
                print("Error: Intergrator Method not found!")
                return        
        return t_array, x_array

    def RK1(self, p: Parameters, x: np.ndarray, t: float, dt: float):
        # update R and U together
        Dx = Derivative_Function(p, x, t)
        x_new = x + dt*Dx
        return x_new

    def RK2(self, p: Parameters, x: np.ndarray, t: float, dt: float):
        Dx1 = Derivative_Function(p, x, t)
        Dx2 = Derivative_Function(p, x + dt*Dx1, t+dt)
        x_new = x + dt/2*(Dx1+Dx2)
        return x_new

    def RK3(self, p: Parameters, x: np.ndarray, t: float, dt: float):
        Dx1 = Derivative_Function(p, x, t)
        Dx2 = Derivative_Function(p, x + dt/2*Dx1, t+dt/2)
        Dx3 = Derivative_Function(p, x - dt*Dx1 + 2*dt*Dx2, t+dt)
        x_new = x + dt/6*(Dx1+4*Dx2+Dx3)
        return x_new

    def RK4(self, p: Parameters, x: np.ndarray, t: float, dt: float):
        Dx1 = Derivative_Function(p, x, t)
        Dx2 = Derivative_Function(p, x + dt/2*Dx1, t+dt/2)
        Dx3 = Derivative_Function(p, x + dt/2*Dx2, t+dt/2)
        Dx4 = Derivative_Function(p, x + dt*Dx3, t+dt)
        x_new = x + dt/6*(Dx1+2*Dx2+2*Dx3+Dx4)
        return x_new

    def RK4_2(self, p: Parameters, x: np.ndarray, t: float, dt: float):
        Dx1 = Derivative_Function(p, x, t)
        Dx2 = Derivative_Function(p, x + dt/2*Dx1, t+dt/2)
        Dx3 = Derivative_Function(p, x + dt*(np.sqrt(2)-1)/2*Dx1 + dt*(1-np.sqrt(2)/2)*Dx2, t+dt/2)
        Dx4 = Derivative_Function(p, x - dt*np.sqrt(2)/2*Dx2 + dt*(1+np.sqrt(2)/2)*Dx3 , t+dt)
        x_new = x + dt/6*(Dx1+2*Dx2+2*Dx3+Dx4)
        return x_new


class Plot_Result:
    def __init__(self, t_array, x_array, Plot_Method: str = "t_x"):
        self.t_array = t_array
        self.x_array = x_array
        self.Plot_Method = Plot_Method

    def plot_result(self):
        if self.Plot_Method == "t_x":
            self.plot_result_t_x(self.t_array, self.x_array)
        elif self.Plot_Method == "fft":
            self.plot_result_fft(self.t_array, self.x_array)
        elif self.Plot_Method == "stft":
            self.plot_result_stft(self.t_array, self.x_array)
        elif self.Plot_Method == "lissajous":
            self.plot_result_lissajous(self.t_array, self.x_array)
        else:
            print("Error: Plot Method not found!")
            return

    def plot_result_t_x(self, t_array, x_array):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(t_array, x_array[:, 0], label="R")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("R(t)")
        ax[0].set_title("R(t) and U(t)")
        ax[0].legend()
        ax[1].plot(t_array, x_array[:, 1], label="U")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("U(t)")
        ax[1].legend()
        plt.savefig("cal1_Bubble_Vibration/t_x.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_fft(self, t_array, x_array):
        plt.figure()
        fs = 1/(t_array[1]-t_array[0])
        f_fft = np.fft.fftfreq(len(x_array[:, 1]), d=1/fs)
        x_fft = np.fft.fft(x_array[:, 1])
        plt.plot(f_fft[:len(f_fft)//2], np.abs(x_fft[:len(f_fft)//2]), label="R")
        x_max = np.max(np.abs(x_fft[:len(f_fft)//2]))
        f_max = f_fft[np.argmax(np.abs(x_fft[:len(f_fft)//2]))]
        plt.scatter(f_max, x_max, label=f"Main Freq = {f_max/1e6:.3f} MHz")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.title("FFT of U(t)")
        plt.savefig("cal1_Bubble_Vibration/fft.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_stft(self, t_array, x_array):
        from scipy.signal import stft
        f, t, Zxx = stft(x_array[:, 1], fs=1/(t_array[1]-t_array[0]), nperseg=100)
        plt.figure()
        plt.pcolormesh(t, f, np.abs(Zxx))
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("STFT of U(t)")
        plt.savefig("cal1_Bubble_Vibration/stft.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_lissajous(self, t_array, x_array):
        plt.figure()
        plt.plot(x_array[:, 0], x_array[:, 1], label="Lissajous")
        plt.xlabel("R")
        plt.ylabel("U")
        plt.legend()
        plt.title("Lissajous Figure of R(t) and U(t)")
        plt.savefig("cal1_Bubble_Vibration/lissajous.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        
    

def main():
    t_start = 0.0
    t_end = 1e-5
    dt = 1e-9
    p = Parameters(R0 = 1.5e-6, omega = 5e6)
    Intergrator_Method = "RK4" # RK1/RK2/RK3/RK4/RK4_2
    Plot_Method = "stft" # t_x/fft/stft/lissajous
    
    x0 = np.array([p.R0, 0.0])
    t_array, x_array = Integrator(p, x0, t_start, t_end, dt, Intergrator_Method).integrator()
    Plot_Result(t_array, x_array, Plot_Method).plot_result()

if __name__ == "__main__":
    main()