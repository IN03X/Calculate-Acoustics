# 221840194 Runbang Wang
# cal3_Acoustic_Wave_Solver
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Parameters
class Parameters:
    def __init__(self,
                Equation: str = "",
                Iterator_Method: str = "",
                Plot_Method: str = "",
                f: float = 1000, 
                z_start: float = 0, 
                z_end: float = 100, 
                dz: float = 1e-2,
                x_start: float = 0.0, 
                x_end: float = 20, 
                dx: float = 1e-2,
                source_width: float = 1/5,
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
        self.f = f
        self.z_start = z_start
        self.z_end = z_end
        self.dz = dz
        self.x_start = x_start
        self.x_end = x_end
        self.dx = dx
        self.source_width = source_width

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
    def generate_bnd_absorption(self):
        pass # TODO YJY

# 3. Run Iterator
class Iterator:
    def __init__(self, p: Parameters, 
                bnd_source: np.ndarray,
                bnd: np.ndarray
                ):    
        self.p = p
        self.bnd_source = bnd_source
        self.bnd = bnd
        
    def iterator(self):
        x_steps = int ((self.p.x_end - self.p.x_start) / self.p.dx)
        x_array = np.linspace(self.p.x_start, self.p.x_end, x_steps)
        z_steps = int ((self.p.z_end - self.p.z_start) / self.p.dz)
        z_array = np.linspace(self.p.z_start, self.p.z_end, z_steps)
        # Initialize result_array with bnd_source, bnd
        # result_array: [z_steps,x_steps]
        result_array = np.zeros((z_steps, x_steps), dtype=complex)
        result_array[0, :] = self.bnd_source
        result_array[1:z_steps-1, 0] = self.bnd[1:z_steps-1, 0]
        result_array[1:z_steps-1, x_steps-1] = self.bnd[1:z_steps-1, 1]
        
        # Calculate result_array
        if self.p.Iterator_Method == "Forward_Euler" and self.p.Equation == "PA":
            rho = self.p.rho
            coef = [ -rho, 1+2*rho, -rho ]
            coef = np.zeros((3,x_steps-2), dtype=complex)
            coef[0,:] = -rho
            coef[1,:] = 1+2*rho
            coef[2,:] = -rho
            for i in range(1, z_steps):
                result_array[i, 1:x_steps-1] = self.Forward_Euler(self.p, result_array[i-1, :], coef=coef)
                result_array[i, 0] = self.bnd[i, 0]
                result_array[i, x_steps-1] = self.bnd[i, 1]

        elif self.p.Iterator_Method == "Dufort_Frankel" and self.p.Equation == "PA":
            rho = self.p.rho
            coef = np.zeros((3,x_steps-2), dtype=complex)
            coef[0,:] = (1-2*rho)/(1+2*rho)
            coef[1,:] = (2*rho)/(1+2*rho)
            coef[2,:] = (2*rho)/(1+2*rho)
            result_array[1,1:x_steps-1] = self.Forward_Euler(self.p, result_array[0, :], coef=coef)
            result_array[1, 0] = self.bnd[1, 0]
            result_array[1, x_steps-1] = self.bnd[1, 1]
            for i in range(2, z_steps):
                result_array[i, 1:x_steps-1] = self.Dufort_Frankel(self.p, result_array[i-1, :], result_array[i-2, :], coef=coef)
                result_array[i, 0] = self.bnd[i, 0]
                result_array[i, x_steps-1] = self.bnd[i, 1]

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
                result_array[i, 0] = self.bnd[i, 0]
                result_array[i, x_steps-1] = self.bnd[i, 1]

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
            result_array[1, 0] = self.bnd[1, 0]
            result_array[1, x_steps-1] = self.bnd[1, 1]
            for i in range(2, z_steps):
                result_array[i, 1:x_steps-1] = self.Dufort_Frankel(self.p, result_array[i-1, :], result_array[i-2, :], coef=coef)
                result_array[i, 0] = self.bnd[i, 0]
                result_array[i, x_steps-1] = self.bnd[i, 1]

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
               result_array[i, 0] = self.bnd[i, 0]
               result_array[i, x_steps-1] = self.bnd[i, 1]
            
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
        title = f"Piston Source @ z=0, f={self.p.f}Hz, {self.p.Iterator_Method}"
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
        pass # TODO HXY
    

def main():
    Equation = "PACC" # PA, PACC: Paraxial Approximation, Paraxial Approximation in Cylindrical Coordinates
    Iterator_Method = "Dufort_Frankel" # Forward_Euler, Dufort_Frankel,Backward_Euler # 221840194 Runbang Wang: 经典显式, Dufort_Frankel显式, 经典隐式
    Plot_Method = "z_x" # z_x, z_xat0
    p = Parameters(Equation=Equation, Iterator_Method=Iterator_Method, Plot_Method=Plot_Method)
    bnd_source = Boundary(p).generate_bnd_source()
    bnd_reflection = Boundary(p).generate_bnd_reflection()

    result_array = Iterator(p=p, bnd_source=bnd_source, bnd=bnd_reflection).iterator()
    
    Plot_Result(p=p, result_array=result_array).plot_result()


if __name__ == "__main__":
    main()
