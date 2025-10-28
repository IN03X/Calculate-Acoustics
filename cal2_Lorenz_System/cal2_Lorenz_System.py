# 221840194 Runbang Wang
# cal2_Lorenz_System.py
import numpy as np
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self,
                 beta: float = 4, 
                 rho: float = 45.92, 
                 sigma: float = 16, 
                 ):        
        self.beta = beta
        self.rho = rho
        self.sigma = sigma


def Derivative_Function(p: Parameters, x: np.ndarray, t: float):
    # x contains R and U
    # Dx contains dR_dt and dU_dt
    xx = x[0]
    y = x[1]
    z = x[2]

    dx_dt = p.sigma*(y-xx)
    dy_dt = p.rho*xx - y - xx*z
    dz_dt = xx*y - p.beta*z

    Dx = np.array([dx_dt, dy_dt, dz_dt])
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
        elif self.Plot_Method == "xyz":
            self.plot_result_xyz(self.t_array, self.x_array)
        else:
            print("Error: Plot Method not found!")
            return

    def plot_result_t_x(self, t_array, x_array):
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        ax[0].plot(t_array, x_array[:, 0], label="x")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("x(t)")
        ax[0].legend()
        ax[1].plot(t_array, x_array[:, 1], label="y")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("y(t)")
        ax[1].legend()
        ax[2].plot(t_array, x_array[:, 2], label="z")
        ax[2].set_xlabel("t")
        ax[2].set_ylabel("z(t)")
        ax[2].legend()
        plt.suptitle("x(t), y(t) and z(t)")
        plt.savefig("cal2_Lorenz_System/t_x.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_fft(self, t_array, x_array):
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        for i in range(3):
            fs = 1/(t_array[1]-t_array[0])
            f_fft = np.fft.fftfreq(len(x_array[:, i]), d=1/fs)
            x_fft = np.fft.fft(x_array[:, i])
            ax[i].plot(f_fft[:len(f_fft)//2], np.abs(x_fft[:len(f_fft)//2]), label=f"Comp {i}")
            x_max = np.max(np.abs(x_fft[:len(f_fft)//2]))
            f_max = f_fft[np.argmax(np.abs(x_fft[:len(f_fft)//2]))]
            ax[i].scatter(f_max, x_max, label=f"Main Freq = {f_max/1e6:.3f} MHz")
            ax[i].set_xlabel("Frequency")
            ax[i].set_ylabel("Magnitude")
            ax[i].legend()
        plt.suptitle("FFT of x(t), y(t), z(t)")
        plt.savefig("cal2_Lorenz_System/fft.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_stft(self, t_array, x_array):
        from scipy.signal import stft
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        for i in range(3):
            f, t, Zxx = stft(x_array[:, i], fs=1/(t_array[1]-t_array[0]), nperseg=100)
            ax[i].pcolormesh(t, f, np.abs(Zxx))
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Frequency")
        plt.suptitle("STFT of x(t), y(t), z(t)")
        plt.savefig("cal2_Lorenz_System/stft.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_lissajous(self, t_array, x_array):
        fig, ax = plt.subplots(3, 1)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i in range(3):
            a, b = pairs[i]
            ax[i].plot(x_array[:, a], x_array[:, b], label=f"Lissajous {a}-{b}")
            ax[i].set_xlabel(f"Comp {a}")
            ax[i].set_ylabel(f"Comp {b}")
            ax[i].legend()
        plt.suptitle("Lissajous Figures of Components")
        plt.savefig("cal2_Lorenz_System/lissajous.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    def plot_result_xyz(self, t_array, x_array):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title("3D Trajectory of x(t), y(t), z(t)")
        plt.savefig("cal2_Lorenz_System/xyz.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        
    
def Lyapunov_Estimation(p: Parameters, x0: np.ndarray, t_start: float, t_end: float, dt: float, Intergrator_Method: str = "RK4",
                        delta_x0: np.ndarray = np.array([0.01, 0.01, 0.01])):
    t_array, x_array = Integrator(p, x0, t_start, t_end, dt, Intergrator_Method).integrator()
    x0_delta = x0 + delta_x0
    t_array_delta, x_array_delta = Integrator(p, x0_delta, t_start, t_end, dt, Intergrator_Method).integrator()
    E = np.linalg.norm(x_array_delta - x_array, axis=1)
    E_valid_idx = E > 0
    lamda = np.mean(np.log(E[E_valid_idx]) / dt)
    print(f"Estimated Lyapunov Exponent: {lamda:.3f}")
    return lamda


def main():
    t_start = 0.0
    t_end = 15
    dt = 1e-3
    p = Parameters()
    Intergrator_Method = "RK4" # RK1/RK2/RK3/RK4/RK4_2
    Plot_Method = "xyz" # t_x/fft/stft/lissajous/xyz
    
    x0 = np.array([1, 1, 1])
    delta_x0 = np.array([0.01, 0.01, 0.01])
    t_array, x_array = Integrator(p, x0, t_start, t_end, dt, Intergrator_Method).integrator()
    Plot_Result(t_array, x_array, Plot_Method).plot_result()
    Lyapunov_Estimation(p, x0, t_start, t_end, dt, Intergrator_Method, delta_x0)


if __name__ == "__main__":
    main()