# 221840194 Runbang Wang
# cal3_Acoustic_Wave_Solver
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.special import j0, jn_zeros
from scipy.integrate import simpson
from scipy.special import j0, j1, jn_zeros
from scipy.integrate import simps


# 1. Define Parameters
class Parameters:
    def __init__(self,
                Equation: str = "",
                Iterator_Method: str = "",
                Plot_Method: str = "",
                Source_Type: str = "", 
                f: float = 1000, 
                z_start: float = 0, 
                z_end: float = 100, 
                dz: float = 1e-3,
                x_start: float = 0.0, 
                x_end: float = 20, 
                dx: float = 1e-1,
                source_width: float = 1/5,
                z_f = 5,
                dz_ratio = 2,
                dx_ratio = 2
                 ):
        k = 2*np.pi*f/343
        if Equation == "PA":
            self.rho = - (1j*dz) / (2*k*(dx)**2) 
        elif Equation == "PACC":
            self.rho1 = - (1j*dz) / (2*k*(dx)**2) 
            self.rho2 = - (1j*dz) / (2*k*dx*2) 
            if x_start != 0.0:
                print("Error: x_start must be 0!")
                return
        else:
            print("Error: Equation not found!")
            return
        if Iterator_Method != "Forward_Euler" and Iterator_Method != "Backward_Euler" and Iterator_Method != "Dufort_Frankel":
            print("Error: Iterator Method not found!")
            return
        if Plot_Method != "z_x" and Plot_Method != "z_xat0":
            print("Error: Plot Method not found!")
            return

        self.Equation = Equation
        self.Iterator_Method = Iterator_Method
        self.Plot_Method = Plot_Method
        self.Source_Type = Source_Type
        self.f = f
        self.z_start = z_start
        self.z_end = z_end
        self.dz = dz
        self.x_start = x_start
        self.x_end = x_end
        self.dx = dx
        self.source_width = source_width
        self.k = k
        self.z_f = z_f
        self.dx_ratio = dx_ratio
        self.dz_ratio = dz_ratio
        

# 2. Define Boundary
class Boundary:
    def __init__(self, p: Parameters):
        self.p = p
    def generate_bnd_source(self):
        # bnd_source: [x_steps] @ z==0
        x_steps = int ((self.p.x_end - self.p.x_start) / self.p.dx)
        bnd_source = np.zeros(x_steps, dtype=complex)
        if self.p.Equation == "PA":
            bnd_source[ int(x_steps//2-x_steps*(self.p.source_width/2)) : int(x_steps//2+x_steps*(self.p.source_width/2)) ] = 1
        elif self.p.Equation == "PACC":
            bnd_source[ 0 : int(x_steps*(self.p.source_width)) ] = 1
        return bnd_source 
    def generate_bnd_reflection(self):
        # bnd_reflection: [z_steps,2] @ x==x_start,x==x_end
        z_steps = int ((self.p.z_end - self.p.z_start) / self.p.dz)
        bnd_reflection = np.zeros((z_steps,2), dtype=complex)
        bnd_reflection[:,0] = 0
        bnd_reflection[:,1] = 0
        return bnd_reflection
    def generate_bnd_absorption(self):# do it in the Class: Iterator
        # bnd_absorption: [z_steps,2] @ x==x_start,x==x_end
        # 左边界 u_0 = u_1 / (1 + i k dx)
        # 右边界 u_N = u_{N-1} / (1 - i k dx)
        # or u_0 = u_1 , u_N = u_{N-1}
        return None

    def generate_focus_source(self):
        # bnd_source: [x_steps] @ z==0
        x_steps = int ((self.p.x_end - self.p.x_start) / self.p.dx)
        focus_source = np.zeros(x_steps, dtype=complex)
        
        if self.p.Equation == "PA":
            start = int(x_steps//2-x_steps*(self.p.source_width/2))
            end = int(x_steps//2+x_steps*(self.p.source_width/2))
            idx = np.arange(start, end)
            x_pos = self.p.x_start + idx * self.p.dx
            delta = np.sqrt( (x_pos-x_steps//2 * self.p.dx)**2 + (self.p.z_f)**2 ) - self.p.z_f
            focus_source[ start : end ] = np.exp(1j* self.p.k * delta)
        elif self.p.Equation == "PACC":
            start = 0
            end = int(x_steps*(self.p.source_width))
            idx = np.arange(start, end)
            x_pos = self.p.x_start + idx * self.p.dx
            delta = np.sqrt( x_pos**2 + (self.p.z_f)**2 ) - self.p.z_f
            focus_source[ start : end ] = np.exp(1j* self.p.k * delta) 
        return focus_source

# 3. Run Iterator
class Iterator:
    def __init__(self, p: Parameters, 
                bnd_source: np.ndarray,
                bnd: np.ndarray
                ):    
        self.p = p
        self.bnd_source = bnd_source
        self.bnd = bnd
    
    def apply_boundary(self, result_array: np.ndarray, i: int):

        x_steps = result_array.shape[1]  # result_array: [z_steps,x_steps]
        if self.p.Equation == "PA":
            if self.bnd is not None:
                result_array[i, 0] = self.bnd[i, 0] #左边界
                result_array[i, x_steps-1] = self.bnd[i, 1] #x_steps-1：x 方向最后一个点（右边界）
            elif self.bnd is None:
                #result_array[i, 0] = result_array[i, 1] / (1 + 1j * self.p.k * self.p.dx)
                #result_array[i, x_steps-1] = result_array[i, x_steps-2] / (1 - 1j * self.p.k * self.p.dx)
                result_array[i, 0] = result_array[i, 1] 
                result_array[i, x_steps-1] = result_array[i, x_steps-2] 
        elif self.p.Equation == "PACC":
            if self.bnd is not None:
                result_array[i, 0] = result_array[i-1, 0]+2*self.p.rho1*(result_array[i-1, 1]-result_array[i-1, 0])
                result_array[i, x_steps-1] = self.bnd[i, 1]
            else:
                result_array[i, 0] = result_array[i-1, 0]+2*self.p.rho1*(result_array[i-1, 1]-result_array[i-1, 0])
                #result_array[i, x_steps-1] = result_array[i, x_steps-2] / (1 - 1j * self.p.k * self.p.dx)
                result_array[i, x_steps-1] = result_array[i, x_steps-2]
            return None
        return None
    
    def iterator(self):
        x_steps = int ((self.p.x_end - self.p.x_start) / self.p.dx)
        x_array = np.linspace(self.p.x_start, self.p.x_end, x_steps)
        z_steps = int ((self.p.z_end - self.p.z_start) / self.p.dz)
        z_array = np.linspace(self.p.z_start, self.p.z_end, z_steps)
        # Initialize result_array with bnd_source, bnd
        # result_array: [z_steps,x_steps]
        result_array = np.zeros((z_steps, x_steps), dtype=complex)
        result_array[0, :] = self.bnd_source
        if self.bnd is not None:
            result_array[1:z_steps-1, 0] = self.bnd[1:z_steps-1, 0]
            result_array[1:z_steps-1, x_steps-1] = self.bnd[1:z_steps-1, 1]
        
        # Calculate result_array
        if self.p.Iterator_Method == "Forward_Euler" and self.p.Equation == "PA":
            rho = self.p.rho
            coef = [ rho, 1-2*rho, rho ]
            coef = np.zeros((3,x_steps-2), dtype=complex)
            coef[0,:] = rho
            coef[1,:] = 1-2*rho
            coef[2,:] = rho
            for i in range(1, z_steps):
                result_array[i, 1:x_steps-1] = self.Forward_Euler(self.p, result_array[i-1, :], coef=coef)
                self.apply_boundary(result_array, i)

        elif self.p.Iterator_Method == "Dufort_Frankel" and self.p.Equation == "PA":
            rho = self.p.rho
            coef = np.zeros((3,x_steps-2), dtype=complex)
            coef[0,:] = (1-2*rho)/(1+2*rho)
            coef[1,:] = (2*rho)/(1+2*rho)
            coef[2,:] = (2*rho)/(1+2*rho)
            result_array[1,1:x_steps-1] = self.Forward_Euler(self.p, result_array[0, :], coef=coef)
            self.apply_boundary(result_array, 1)

            for i in range(2, z_steps):
                result_array[i, 1:x_steps-1] = self.Dufort_Frankel(self.p, result_array[i-1, :], result_array[i-2, :], coef=coef)
                self.apply_boundary(result_array, i)


        elif self.p.Iterator_Method == "Forward_Euler" and self.p.Equation == "PACC":
            rho1 = self.p.rho1
            rho2 = self.p.rho2
            coef = np.zeros((3,x_steps-2), dtype=complex)
            r = x_array[1:x_steps-1]
            coef[0,:] = np.where(np.abs(r)<1e-8, rho1, rho1 + rho2/r)
            coef[1,:] = 1-2*rho1
            coef[2,:] = np.where(np.abs(r)<1e-8, rho1, rho1 - rho2/r)
            for i in range(1, z_steps):
                result_array[i, 1:x_steps-1] = self.Forward_Euler(self.p, result_array[i-1, :], coef=coef)
                self.apply_boundary(result_array, i)

        elif self.p.Iterator_Method == "Dufort_Frankel" and self.p.Equation == "PACC":
            rho1 = self.p.rho1
            rho2 = self.p.rho2
            coef = np.zeros((3,x_steps-2), dtype=complex)
            r = x_array[1:x_steps-1]
            # r = np.abs(x_array[1:x_steps-1])
            coef[0,:] = (1-2*rho1)/(1+2*rho1)
            coef[1,:] = np.where(np.abs(r)<1e-8, (2*rho1)/(1+2*rho1), (2*rho1-2*rho2/r)/(1+2*rho1))
            coef[2,:] = np.where(np.abs(r)<1e-8, (2*rho1)/(1+2*rho1), (2*rho1+2*rho2/r)/(1+2*rho1))
            result_array[1,1:x_steps-1] = self.Forward_Euler(self.p, result_array[0, :], coef=coef)
            self.apply_boundary(result_array, 1)
            for i in range(2, z_steps):
                result_array[i, 1:x_steps-1] = self.Dufort_Frankel(self.p, result_array[i-1, :], result_array[i-2, :], coef=coef)
                self.apply_boundary(result_array, i)

        elif self.p.Iterator_Method == "Backward_Euler" and self.p.Equation == "PA":
            pass # TODO HPC
            rho = self.p.rho
            a = np.ones(x_steps-2,dtype=complex)*(1+2*rho)
            b = -np.ones(x_steps-2,dtype=complex)*rho
            c = -np.ones(x_steps-2,dtype=complex)*rho
            l = np.zeros(x_steps-2,dtype=complex)
            u = np.zeros(x_steps-2,dtype=complex)
            u[0] = a[0]
            for i in range(1,x_steps-2):
                l[i] = b[i]/u[i-1]
                u[i] = a[i]-l[i]*c[i-1]
            for i in range(1, z_steps):
                d = result_array[i-1, 1:x_steps-1]
               
                d[0] = d[0]+rho* self.bnd[i, 0]
                d[x_steps-3] = d[x_steps-3]+rho*self.bnd[i, 1]
                result_array[i, 1:x_steps-1] = self.Backward_Euler(self.p, d, l , u, c)
                #result_array[i, 1:x_steps-1] = self.Gauss_Euler(self.p, d, a, b, c)
                self.apply_boundary(result_array, i)
            
        elif self.p.Iterator_Method == "Backward_Euler" and self.p.Equation == "PACC":
            pass # TODO HPC
            rho1 = self.p.rho1
            rho2 = self.p.rho2
            r = x_array[1:x_steps]
            a = np.zeros(x_steps-1,dtype=complex)
            b = -np.ones(x_steps-1,dtype=complex)*rho1
            c = -np.zeros(x_steps-1,dtype=complex)
            l = np.zeros(x_steps-1,dtype=complex)
            u = np.zeros(x_steps-1,dtype=complex)
            a[0] = 1+2*rho1
            c[0] = -2*rho1
            a[1:] = 1+2*rho1+2*rho2/r[0:x_steps-2]
            c[1:] = -rho1-2*rho2/r[0:x_steps-2]
            u[0] = a[0]
            for i in range(1,x_steps-1):
                l[i] = b[i]/u[i-1]
                u[i] = a[i]-l[i]*c[i-1]
            for i in range(1, z_steps):
               d = result_array[i-1, 0:x_steps-1];
               d[x_steps-2] = d[x_steps-2]-c[x_steps-2]*self.bnd[i, 1]
               result_array[i, 0:x_steps-1] = self.Backward_Euler(self.p, d, l , u, c)
               #result_array[i, 0:x_steps-1] = self.Gauss_Euler(self.p, d, a, b, c)
               result_array[i, x_steps-1] = self.bnd[i, 1]
        return result_array

    def Forward_Euler(self, p: Parameters, x: np.ndarray, coef: np.ndarray):
        x_steps = x.size
        x_new = np.zeros(x_steps, dtype=complex)
        x_new[1:x_steps-1] = coef[0,:]*x[0:x_steps-2] + coef[1,:]*x[1:x_steps-1] + coef[2,:]*x[2:x_steps]
        return x_new[1:x_steps-1]
    def Dufort_Frankel(self, p: Parameters, x: np.ndarray, x_old: np.ndarray, coef: np.ndarray):
        x_steps = x.size
        x_new = np.zeros(x_steps, dtype=complex)
        x_new[1:x_steps-1] = coef[0,:]*x_old[1:x_steps-1] + coef[1,:]*x[0:x_steps-2] + coef[2,:]*x[2:x_steps]
        return x_new[1:x_steps-1]
    def Backward_Euler(self, p: Parameters, x: np.ndarray , l , u, c):
        x_steps = x.size
        y = np.zeros(x_steps,dtype=complex)
        y[0] = x[0]
        for i in range(1,x_steps):
            y[i] = x[i]-l[i]*y[i-1]
        x_new = np.zeros(x_steps, dtype=complex)
        x_new[x_steps-1] = y[x_steps-1]/u[x_steps-1]
        for i in range(x_steps-2,-1,-1):
            x_new[i]=(y[i]-c[i]*x_new[i+1])/u[i]
        return x_new
        pass # TODO HPC
    def Gauss_Euler(self, p: Parameters, x: np.ndarray , a , b , c):
        x_steps = x.size
        for i in range(1,x_steps):
            a[i] = a[i]-b[i]*c[i-1]/a[i-1]
            x[i] = x[i]-b[i]*x[i-1]/a[i-1]
        x_new = np.zeros(x_steps, dtype=complex)
        x_new[x_steps-1] = x[x_steps-1]/a[x_steps-1]
        for i in range(x_steps-2,-1,-1):
            x_new[i] = (x[i]-x_new[i+1]*c[i])/a[i]
        return x_new
        pass # TODO HPC

# 4. Plot Result
class Plot_Result:
    def __init__(self, p: Parameters, result_array: np.ndarray):
        self.result_array = result_array
        self.Plot_Method = p.Plot_Method 
        self.p = p
        
    def plot_result(self):
        if self.Plot_Method == "z_x":
            self.plot_result_z_x(self.result_array)
        elif self.Plot_Method == "z_xat0":
            self.plot_result_z_x0(self.result_array)

    def plot_result_z_x(self, result_array):
        z_steps = result_array.shape[0]
        x_steps = result_array.shape[1]
        z_array = np.linspace(self.p.z_start, self.p.z_end, z_steps)
        x_array = np.linspace(self.p.x_start, self.p.x_end, x_steps)
        plt.figure()
        plt.imshow(np.abs(result_array), extent=[x_array[0], x_array[-1], z_array[0], z_array[-1]], aspect='auto', origin='lower', cmap='jet')
        if self.p.Equation == "PA":
            xlabel = 'y/m'
        elif self.p.Equation == "PACC":
            xlabel = 'r/m'
        plt.xlabel(xlabel=xlabel)
        plt.ylabel('z/m')
        title = f"{self.p.Source_Type} Source @ z=0, f={self.p.f}Hz, {self.p.Iterator_Method}"
        plt.title(title, fontdict={'fontsize': 10, 'fontweight': 'bold'})
        plt.colorbar()
        plt.savefig(f'{self.p.Equation}_{self.p.Iterator_Method}_{self.p.Plot_Method}.pdf')
        plt.show()  
    def plot_result_z_x0(self, result_array):
        #plot x=0
        z_steps = result_array.shape[0]
        x_steps = result_array.shape[1]
        z_array = np.linspace(self.p.z_start, self.p.z_end, z_steps)
        plt.figure()
        if self.p.Equation == "PA":
            plt.plot(z_array, np.abs(result_array[:,x_steps//2]))
        elif self.p.Equation == "PACC":
            plt.plot(z_array, np.abs(result_array[:,0]))
        plt.xlabel('z/m')
        plt.ylabel('Abs(x=0)')
        title = f"Pressure @ x=0, f={self.p.f}Hz, {self.p.Iterator_Method}"
        plt.title(title, fontdict={'fontsize': 10, 'fontweight': 'bold'})
        plt.savefig(f'{self.p.Equation}_{self.p.Iterator_Method}_{self.p.Plot_Method}.pdf')
        plt.show()

        

# 5. Evaluation 

class Evaluation:
    def __init__(self, p: Parameters, result_array: np.ndarray):
        self.p = p
        self.result_array = result_array
        
    def evaluate_result(self):
        pass
    
    def compute_L2_error(self, coarse_result, dz_ratio, dx_ratio, plot=True):
        
        fine_result = self.result_array
        if not (
            (isinstance(dx_ratio, int) or (isinstance(dx_ratio, float) and dx_ratio.is_integer()))
            and
            (isinstance(dz_ratio, int) or (isinstance(dz_ratio, float) and dz_ratio.is_integer()))
        ):
            raise ValueError("Step ratios must be integers for direct comparison")
        
        dz_ratio = int(dz_ratio)
        dx_ratio = int(dx_ratio)        
        fine_downsampled = fine_result[::dz_ratio, ::dx_ratio]
        min_z = min(fine_downsampled.shape[0], coarse_result.shape[0])
        min_x = min(fine_downsampled.shape[1], coarse_result.shape[1])
        fine_downsampled = fine_downsampled[:min_z, :min_x]
        coarse_result = coarse_result[:min_z, :min_x]
        amp_fine = np.abs(fine_downsampled)
        amp_coarse = np.abs(coarse_result)
        amp_error_map = amp_fine - amp_coarse
        amp_error = np.sqrt(np.mean(amp_error_map**2))
        phase_fine = np.angle(fine_downsampled)
        phase_coarse = np.angle(coarse_result)
        phase_diff = phase_fine - phase_coarse
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_error_map = phase_diff
        phase_error = np.sqrt(np.mean(phase_diff**2))
        if plot:
            self.plot_error_distribution(amp_error_map, phase_error_map, dz_ratio, dx_ratio)
        
        return amp_error, phase_error
    
    def plot_error_distribution(self, amp_error_map, phase_error_map, dz_ratio, dx_ratio):
        z_steps = amp_error_map.shape[0]
        x_steps = amp_error_map.shape[1]
        z_coords = np.linspace(self.p.z_start, self.p.z_end, z_steps)
        x_coords = np.linspace(self.p.x_start, self.p.x_end, x_steps)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(amp_error_map, 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude Error')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title(f'Amplitude Error Distribution (dz_ratio={dz_ratio}, dx_ratio={dx_ratio})')
        
        '''
        plt.subplot(2, 2, 2)
        plt.imshow(phase_error_map, 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='plasma')
        plt.colorbar(label='Phase Error (rad)')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title(f'Phase Error Distribution (dz_ratio={dz_ratio}, dx_ratio={dx_ratio})')
        '''
        mean_amp_error_z = np.mean(np.abs(amp_error_map), axis=1)
        mean_phase_error_z = np.mean(np.abs(phase_error_map), axis=1)        
        plt.subplot(1, 2, 2)
        plt.plot(z_coords, mean_amp_error_z)
        plt.xlabel('z/m')
        plt.ylabel('Mean Amplitude Error')
        plt.title('Mean Amplitude Error Along Depth')
        plt.grid(True) 

        '''
        plt.subplot(2, 2, 4)
        plt.plot(z_coords, mean_phase_error_z)
        plt.xlabel('z/m')
        plt.ylabel('Mean Phase Error (rad)')
        plt.title('Mean Phase Error Along Depth')
        plt.grid(True)
        '''
        plt.tight_layout()
        plt.savefig(f'Error_Analysis_dz_{dz_ratio}_dx_{dx_ratio}.pdf')
        plt.show()
               
    def analytical_solution(self, z_coords, x_coords, source_type="piston"):
        k = 2*np.pi*self.p.f/343
        w0 = self.p.source_width
        Z, X = np.meshgrid(z_coords, x_coords, indexing='ij')
        
        if self.p.Equation == "PA":
            if source_type == "piston":
                return self._pa_piston_analytical(Z, X, k, w0)
            else:  # focus
                return self._pa_focus_analytical(Z, X, k, w0)
        elif self.p.Equation == "PACC":
            if source_type == "piston":
                return self._pacc_piston_analytical(Z, X, k, w0)
            else:  # focus
                return self._pacc_focus_analytical(Z, X, k, w0)
        else:
            raise ValueError(f"Unsupported equation type: {self.p.Equation}")
    
    def _pa_piston_analytical(self, Z, X, k, w0):
        return self._angular_spectrum_method(Z, X, k, w0, "piston")
    
    def _pa_focus_analytical(self, Z, X, k, w0):
        return self._angular_spectrum_method(Z, X, k, w0, "focus")
    
    def _pacc_piston_analytical(self, Z, X, k, w0):
        return self.hankel_transform_method(Z, X, k, w0, "piston")
    
    def _pacc_focus_analytical(self, Z, X, k, w0):
        return self.hankel_transform_method(Z, X, k, w0, "focus")
    
    def _angular_spectrum_method(self, Z, X, k, w0, source_type):
        x_coords = X[0, :]
        z_coords = Z[:, 0]
        
        dx = x_coords[1] - x_coords[0]
        Nx = len(x_coords)
        fx = fftfreq(Nx, dx)
        
        if source_type == "piston":
            p0 = np.zeros(Nx, dtype=complex)
            start_idx = int(Nx//2 - Nx*(w0/2))
            end_idx = int(Nx//2 + Nx*(w0/2))
            p0[start_idx:end_idx] = 1.0
        else:  # focus
            p0 = np.zeros(Nx, dtype=complex)
            start_idx = int(Nx//2 - Nx*(w0/2))
            end_idx = int(Nx//2 + Nx*(w0/2))
            x_pos = x_coords[start_idx:end_idx]
            center_x = x_coords[Nx//2]
            delta = np.sqrt((x_pos - center_x)**2 + self.p.z_f**2) - self.p.z_f
            p0[start_idx:end_idx] = np.exp(1j * k * delta)
        
        P0 = fft(p0)
        
        p_analytical = np.zeros(Z.shape, dtype=complex)
        for i, z in enumerate(z_coords):
            if z == 0:
                p_analytical[i, :] = p0
            else:
                k_x = 2 * np.pi * fx
                k_z = k - k_x**2 / (2 * k)
                H = np.exp(-1j * k_z * z)
                P_z = P0 * H
                p_analytical[i, :] = ifft(P_z)
        
        return p_analytical
        
    def _hankel_transform(self, f, r, kr):
        n = len(r)
        m = len(kr)
        
        R, KR = np.meshgrid(r, kr)
        
        integrand = f * R * j0(KR * R)
        
        F = np.zeros_like(kr, dtype=complex)
        for i in range(m):
            F[i] = simps(integrand[i, :], r)
        
        return F
    
    def _inverse_hankel_transform(self, F, kr, r):
        n = len(r)
        m = len(kr)
        
        KR, R = np.meshgrid(kr, r)
        integrand = F * KR * j0(KR * R)
        
        f = np.zeros_like(r, dtype=complex)
        for j in range(n):
            f[j] = simps(integrand[:, j], kr)        
        return f
    
    def discrete_hankel_transform(self, f, r, kr_max=None):
        n = len(r)        
        alpha = jn_zeros(0, n)        
        R = r[-1]        
        if kr_max is None:
            kr_max = alpha[-1] / R
        kr = alpha / R        
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                T[i, j] = j0(alpha[i] * r[j] / R)
        
        weights = 2 / (R**2 * j1(alpha)**2)        
        F = weights * T @ (f * r)        
        return kr, F
    
    def inverse_discrete_hankel_transform(self, F, kr, r):        
        n = len(r)
        alpha = jn_zeros(0, n)
        R = r[-1] 
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                T[i, j] = j0(alpha[i] * r[j] / R)    
        weights = R**2 / (2 * j1(alpha)**2)    
        f = weights * T @ F    
        return f
    
    def hankel_transform_method(self, Z, R, k, w0, source_type):    
        r_coords = R[0, :]
        z_coords = Z[:, 0]  
        p0 = self._get_initial_field(r_coords, w0, source_type, k) 
        kr, P0 = self.discrete_hankel_transform(p0, r_coords)
        p_analytical = np.zeros(Z.shape, dtype=complex)
        for i, z in enumerate(z_coords):    
            if i % max(1, len(z_coords)//1000) == 0:  # 每10%打印一次
                print(f"Progress: {i/len(z_coords)*100:.1f}%")
            kz = k - kr**2 / (2 * k)  
            H = np.exp(-1j * kz * z)  
            P_z = P0 * H    
            p_z = self.inverse_discrete_hankel_transform(P_z, kr, r_coords)
            p_analytical[i, :] = p_z
        return p_analytical
    
    def _get_initial_field(self, r, w0, source_type, k):
        if source_type == "piston":
            p0 = np.zeros_like(r, dtype=complex)
            p0[r <= w0] = 1.0
        else:  # focus
            p0 = np.zeros_like(r, dtype=complex)
            mask = r <= w0
            delta = np.sqrt(r[mask]**2 + self.p.z_f**2) - self.p.z_f
            p0[mask] = np.exp(1j * k * delta)
        return p0
    
    def compute_L2_error_analytical(self, source_type="piston", plot=True):
        p_numerical = self.result_array
        z_steps, x_steps = p_numerical.shape
        z_coords = np.linspace(self.p.z_start, self.p.z_end, z_steps)
        x_coords = np.linspace(self.p.x_start, self.p.x_end, x_steps)
        p_analytical = self.analytical_solution(z_coords, x_coords, source_type)
        
        
        if p_analytical.shape != p_numerical.shape:
            from scipy.interpolate import interp2d
            z_analytical = np.linspace(self.p.z_start, self.p.z_end, p_analytical.shape[0])
            x_analytical = np.linspace(self.p.x_start, self.p.x_end, p_analytical.shape[1])
            
            interp_real = interp2d(x_analytical, z_analytical, np.real(p_analytical))
            interp_imag = interp2d(x_analytical, z_analytical, np.imag(p_analytical))
            
            p_analytical_real = interp_real(x_coords, z_coords)
            p_analytical_imag = interp_imag(x_coords, z_coords)
            p_analytical = p_analytical_real + 1j * p_analytical_imag
        
        amp_numerical = np.abs(p_numerical)
        amp_analytical = np.abs(p_analytical)
        amp_error_map = amp_numerical - amp_analytical
        amp_error = np.sqrt(np.mean(amp_error_map**2))
        
        phase_numerical = np.angle(p_numerical)
        phase_analytical = np.angle(p_analytical)
        
        phase_diff = phase_numerical - phase_analytical
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_error_map = phase_diff
        phase_error = np.sqrt(np.mean(phase_diff**2))
        
        if plot:
            self.plot_error_distribution_analytical(
                amp_error_map, phase_error_map, 
                p_numerical, p_analytical,
                z_coords, x_coords, source_type
            )
        
        return amp_error, phase_error, p_analytical
    
    def plot_error_distribution_analytical(self, amp_error_map, phase_error_map, 
                                         p_numerical, p_analytical, z_coords, x_coords, source_type):
        
        
        plt.figure(figsize=(20, 6))     
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(p_numerical), 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title('Numerical Solution (Amplitude)')
        
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(p_analytical), 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title('Analytical Solution (Amplitude)')
        
        
        plt.subplot(1, 3, 3)
        plt.imshow(amp_error_map, 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='RdBu_r')
        plt.colorbar(label='Amplitude Error')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title('Amplitude Error Distribution')
        
        '''
        
        plt.subplot(2, 3, 4)
        plt.imshow(np.angle(p_numerical), 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='hsv')
        plt.colorbar(label='Phase (rad)')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title('Numerical Solution (Phase)')
        
       
        plt.subplot(2, 3, 5)
        plt.imshow(np.angle(p_analytical), 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='hsv')
        plt.colorbar(label='Phase (rad)')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title('Analytical Solution (Phase)')
        
       
        plt.subplot(2, 3, 6)
        plt.imshow(phase_error_map, 
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                  aspect='auto', origin='lower', cmap='RdBu_r')
        plt.colorbar(label='Phase Error (rad)')
        plt.xlabel('x/m')
        plt.ylabel('z/m')
        plt.title('Phase Error Distribution')
        '''
        plt.suptitle(f'Error Analysis - {self.p.Equation} Equation, {self.p.Iterator_Method} Method, {source_type.capitalize()} Source', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'Analytical_Error_Analysis_{self.p.Equation}_{source_type}.pdf')
        plt.show()
        
        
        
        # 沿深度方向的误差(两数值解)
        self.plot_error_along_depth(amp_error_map, phase_error_map, z_coords, source_type)
    
    def plot_error_along_depth(self, amp_error_map, phase_error_map, z_coords, source_type):
        
        mean_amp_error_z = np.mean(np.abs(amp_error_map), axis=1)
        mean_phase_error_z = np.mean(np.abs(phase_error_map), axis=1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 1, 1)
        plt.plot(z_coords, mean_amp_error_z)
        plt.xlabel('z/m')
        plt.ylabel('Mean Amplitude Error')
        plt.title('Mean Amplitude Error Along Depth')
        plt.grid(True)
        '''
        plt.subplot(1, 2, 2)
        plt.plot(z_coords, mean_phase_error_z)
        plt.xlabel('z/m')
        plt.ylabel('Mean Phase Error (rad)')
        plt.title('Mean Phase Error Along Depth')
        plt.grid(True)
        '''
        
        plt.suptitle(f'Error Along Depth - {self.p.Equation} Equation, {source_type.capitalize()} Source', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'Error_Along_Depth_{self.p.Equation}_{source_type}.pdf')
        plt.show()
        

def main():
    Equation = "PACC" # PA, PACC: Paraxial Approximation, Paraxial Approximation in Cylindrical Coordinates
    Iterator_Method = "Backward_Euler" # Forward_Euler, Dufort_Frankel,Backward_Euler # 221840194 Runbang Wang: 经典显式, Dufort_Frankel显式, 经典隐式
    Plot_Method = "z_x" # z_x, z_xat0
    Source_type = "focus" # piston, focus
    p = Parameters(Equation=Equation, Iterator_Method=Iterator_Method, Plot_Method=Plot_Method,Source_Type=Source_type)
    if Source_type == "piston":
        bnd_source = Boundary(p).generate_bnd_source()
    else:  # focus
        bnd_source = Boundary(p).generate_focus_source()
    bnd_reflection = Boundary(p).generate_bnd_reflection()
    # bnd_absorption = Boundary(p).generate_bnd_absorption()

    result_array = Iterator(p=p, bnd_source=bnd_source, bnd=bnd_reflection).iterator()
    
    Plot_Result(p=p, result_array=result_array).plot_result()

    evaluation = Evaluation(p, result_array)
    coarse_params = Parameters(
        Equation=Equation,
        Iterator_Method=Iterator_Method,
        Plot_Method=Plot_Method,
        dz=p.dz * p.dz_ratio,
        dx=p.dx * p.dx_ratio 
        )    
    if Source_type == "piston":
        bnd_source_coarse = Boundary(coarse_params).generate_bnd_source()
    else:  # focus
        bnd_source_coarse = Boundary(coarse_params).generate_focus_source()
    bnd_reflection_coarse = Boundary(coarse_params).generate_bnd_reflection()
    result_array_coarse = Iterator(p=coarse_params, bnd_source=bnd_source_coarse,bnd=bnd_reflection_coarse).iterator()
    Plot_Result(p=coarse_params, result_array=result_array_coarse).plot_result()

    amp_error, phase_error = evaluation.compute_L2_error(
        coarse_result=result_array_coarse,
        dz_ratio=p.dz_ratio,
        dx_ratio=p.dx_ratio,
        plot=True
    )
    print(f"L2 error between fine and coarse grid results({Source_type} source,{Equation} equation),{Iterator_Method} method...")
    print(f"Amplitude L2 Error: {amp_error:.4e}")
    # print(f"Phase L2 Error: {phase_error:.4e} rad")

    print(f"Computing L2 error between numerical and analytical solutions ({Source_type} source,{Equation} equation),{Iterator_Method} method...")
    amp_error_analytical, phase_error_analytical, p_analytical = evaluation.compute_L2_error_analytical(
        source_type=Source_type, plot=True
    )
    print(f"Analytical vs Numerical - Amplitude L2 Error: {amp_error_analytical:.4e}")
    # print(f"Analytical vs Numerical - Phase L2 Error: {phase_error_analytical:.4e} rad")


if __name__ == "__main__":
    main()
