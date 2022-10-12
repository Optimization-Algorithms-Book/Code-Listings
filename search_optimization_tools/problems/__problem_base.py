import abc
import numpy as np
import matplotlib.pyplot as plt


class problem_base(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_init_solution(self):
        pass
    
    @abc.abstractmethod
    def get_neighbour_solution(self, sol):
        pass

    @abc.abstractmethod
    def eval_solution(self, sol):
        pass


class continuous_function_base(problem_base):

    def __init__(self, eval_func, bounds, step=1) -> None:
        super().__init__()
        self.__eval_func = eval_func
        self.__bounds = bounds        
        self.__step = step

    def get_init_solution(self):
        return self.__bounds[:, 0] + np.random.rand(len(self.__bounds)) * (self.__bounds[:, 1] - self.__bounds[:, 0])

    def get_neighbour_solution(self, sol):
        new_sol = self.__bounds[:, 0] - 1
        while (new_sol < self.__bounds[:, 0]).any() or (new_sol > self.__bounds[:, 1]).any():
            new_sol = sol + np.random.randn(len(self.__bounds)) * self.__step
        return new_sol

    def eval_solution(self, sol):
        return self.__eval_func(*sol)

    def plot(self, best_sol, figsize=(12, 10), fontsize=18, save_file=None):
        plt.figure(figsize=figsize)

        if self.__bounds.shape[0] == 1:
            plt.title("Continuous function optimization")
            x = np.arange(self.__bounds[0][0], self.__bounds[0][1], 0.01)
            Z = [self.eval_solution([_]) for _ in x]
            plt.plot(x, Z, label="f(x)")
            plt.plot(best_sol, self.eval_solution(best_sol), 'sr', label="global minimum")
            plt.ylabel("f(x)", fontsize=fontsize)
            plt.xlabel("x", fontsize=fontsize)
            plt.legend(loc='best', fancybox=True, shadow=True, fontsize=fontsize)
            plt.grid()
            print("global minimum: x = %.4f, f(x) = %.4f" % (best_sol, self.eval_solution(best_sol)))

        elif self.__bounds.shape[0] == 2:
            x = np.linspace(self.__bounds[0][0], self.__bounds[0][1], 100)
            y = np.linspace(self.__bounds[1][0], self.__bounds[1][1], 100)
            X, Y = np.meshgrid(x, y)
            Z = np.asarray([[self.__eval_func(X[i][j],Y[i][j]) for i in range(X.shape[0])] for j in range(X.shape[1])])
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
            # ax.ylabel("x_2", fontsize=fontsize)
            # ax.xlabel("x_1", fontsize=fontsize)
            # ax.zlabel("f(x)", fontsize=fontsize)
            ax.scatter(best_sol[0], best_sol[1], self.eval_solution(best_sol))
            ax.legend(loc='best', fancybox=True, shadow=True, fontsize=fontsize)
            ax.grid()
            ax.set_title('Continuous function optimization')
            print("global minimum: x = %.4f, %.4f, f(x) = %.4f" % (*best_sol, self.eval_solution(best_sol)))

        else:
            print("%dd can't be plot"%self.__bounds.shape[0])
        
                
        if save_file:
            plt.savefig(save_file)
