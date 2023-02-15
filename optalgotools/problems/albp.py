from .__problem_base import ProblemBase
import numpy as np
import pandas as pd 
import random as rd
import math
import matplotlib.pyplot as plt
import requests
import io

class ALBP(ProblemBase):
    def __init__(self, url, problem, Cycle_time) -> None:
        super().__init__()
        self.url = url
        self.problem = problem
        self.Cycle_time = Cycle_time
        self.tasks = pd.DataFrame(columns=['Task', 'Duration']); 
        self.Prec = pd.DataFrame(columns=['TASK', 'IMMEDIATE_PRECEDESSOR']); 
        self.Tabu_Structure = pd.DataFrame(columns=['Task_1', 'Task_2', 'Feasible', 'Tabued','Value']); 
        self.total_processing_time = None
        self.N = None
        self.TN_M_WS = None
        self.TN_P_WS = None
        self.WS = None
        self.SI = None
        self.average_WS_time = None
        self.Sol = None
        
        
    def Initial_Solution(self, s):
        sol = np.arange(s)
        Sol_list = list(sol)
        for i in range(len(sol)):
            Sol_list[i] = 'T' + str(Sol_list[i]+1)
        for i in range(len(sol)): 
            x = rd.randint(0, s-1)
            y = rd.randint(0, s-1)
            t1 = Sol_list[x]
            t2 = Sol_list[y]
            Sol_list[x] = t2
            Sol_list[y] = t1
        return Sol_list

    def Make_Solution_Feasible(self, solution, Prec):
        lsol = solution[:]
        for i in range(len(lsol)):
            for j in range(i, len(lsol)):
                if(i<j):
                    if(len((Prec.loc[(Prec['TASK'] == solution[j])]).loc[(Prec['IMMEDIATE_PRECEDESSOR']==solution[i] )])==1): 
                        temp1 = lsol[i] 
                        temp2 = lsol[j] 
                        lsol[i] = temp2 
                        lsol[j] = temp1
                        solution = lsol[:] 
        return lsol

    def get_index(self, s):
        k = s[1:]
        k = int(k)-1
        return k
    
    def Smoothing_index(self, solution2, WS, tasks, show = False):  
        WS_time = np.zeros(len(solution2), float)
        j =0 
        i = 0
        l = 0
        while i < len(solution2):
            y = float(tasks.loc[self.get_index(solution2[i])][tasks.columns[1]]) 
            if WS_time[j] +  y <= self.Cycle_time:
                WS_time[j] = WS_time[j] + y
                i = i +1 
                l = l +1 
            else: 
                j = j+1 
        WS_time = WS_time[WS_time != 0]
        WS_Max = WS_time.max()
        ns = np.zeros(len(tasks), float)
        ns = ((WS_Max- WS_time)**2)/(WS_time.size)
        SI = ns.sum()
        SI = math.sqrt(SI)
        if show == True: 
            print("#"*8, "The Smoothing Index value for {} solution sequence is: {}".format(solution2 ,SI),"#"*8)
            print("#"*8, "The number of workstations for {} solution sequence is: {}".format(solution2 ,WS_time.size),"#"*8)
            print("#"*8, "The workloads of workstation for {} solution sequence are: {}".format(solution2 ,WS_time),"#"*8)
        #return SI
        return WS_time.size

    
    def Create_Tabu_Strcuture(self, tasks, Prec):
        Tabu_Structure = pd.DataFrame(columns=['Task_1', 'Task_2', 'Feasible', 'Tabued','Value']); 
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                if i<j: 
                    p1 = "T"+str(i+1)
                    p2 = "T"+str(j+1)
                    if(len((Prec.loc[(Prec['TASK'] == p1)]).loc[(Prec['IMMEDIATE_PRECEDESSOR']==p2)])==1): 
                        Tabu_Structure.loc[len(Tabu_Structure.index)] = ["T" + str(i+1), "T" + str(j+1), "F", 0, 1000] 
                    else: 
                        Tabu_Structure.loc[len(Tabu_Structure.index)] = ["T" + str(i+1), "T" + str(j+1), "T", 0, 1000] 
        return Tabu_Structure
    
    def N_Swap(self, solution, i1, i2):
        sol = solution[:]; 
        temp1 = sol[i1]
        temp2 = sol[i2]
        sol[i1] = temp2
        sol[i2] = temp1
        sol = self.Make_Solution_Feasible(sol, self.Prec)
        return sol

    def Make_Solution_to_plot(self, solution, WS, tasks, Cycle_time):
        data = {}   
        WS_time = np.zeros(WS, float)
        j =0 
        i = 0
        l = 0
        m = "W"+str(j+1)
        while i < len(solution):
            y = float(tasks.loc[self.get_index(solution[i])][tasks.columns[1]]) 
            if WS_time[j] +  y <= Cycle_time:
                WS_time[j] = WS_time[j] + y
                i = i +1 
                l = l +1 
                data[m] = WS_time[j]
            else: 
                j = j+1 
                m = "W"+str(j+1)
                data[m] = 0           
        WS_Names = list(data.keys())
        WS_Workloads = list(data.values())
        fig = plt.figure()
        plt.axhline(y = Cycle_time, color = 'r', linestyle = '-')
        plt.bar(WS_Names, WS_Workloads, color ='blue', width = 0.4)
        plt.xlabel("Workstations ")
        plt.ylabel("Time")
        plt.title("The total workloads of workstations")
        return plt    
    
    def Get_Tasks(self, url, problem):
        s = 0
        tasks = pd.DataFrame(columns=['Task', 'Duration'])
        response = requests.get(url+problem)
        if response.status_code == 200:
            file = io.StringIO(response.text)
            contents = file.readlines()
            for line in contents: 
                if line.find(",")==-1:
                    s = s+1
                    tasks.loc[len(tasks.index)] = ["T"+str(s),  line.replace("\n","")]
            file.close()
        else:
            print(f"Failed to fetch URL: {response.status_code}")
        return tasks;
   
## To read task duration from a file
#     def Get_Tasks(self, path, problem):
#         s = 0
#         tasks = pd.DataFrame(columns=['Task', 'Duration'])
#         with open(path+"/"+ problem) as f:
#             contents = f.readlines()
#             for line in contents: 
#                 if line.find(",")==-1:
#                     s = s+1
#                     tasks.loc[len(tasks.index)] = ["T"+str(s),  line.replace("\n","")]
#             f.close()
#         #tasks = tasks.drop([s-1])
#         return tasks;


    def Get_Prec(self, url, problem):
        Prec2 = pd.DataFrame(columns=['TASK', 'IMMEDIATE_PRECEDESSOR'])
        response = requests.get(url+problem)
        if response.status_code == 200:
            file = io.StringIO(response.text)
            contents = file.readlines()
            for line in contents: 
                if line.find(",")!=-1:
                    l = line.index(",")
                    s1 = line[0:l]
                    s2 = line[l+1:]
                    s2 = s2.replace("\n","")
                    if s2 !="-1":
                        Prec2.loc[len(Prec2.index)] = ["T"+s1,  "T"+s2]
            file.close()
        #Prec2.loc[len(Prec2.index)] =["T"+str(len(self.tasks.index)-1), "T"+str(len(self.tasks.index))]
        for l in self.tasks.index: 
            t1 = self.tasks.iat[l,0]
            for p in self.tasks.index: 
                if p!= l:
                    t2 = self.tasks.iat[p,0]
                    if(len((Prec2.loc[(Prec2['TASK'] == t1)]).loc[(Prec2['IMMEDIATE_PRECEDESSOR']==t2 )])==1):
                        Prec3 = Prec2[(Prec2['TASK'] == t2)]
                        for u in range(len(Prec3.index)):
                            Prec2.loc[len(Prec2.index)] = [t1,  Prec3.iat[u,1]]
        return Prec2;

## to read precedence froma  file    
#     def Get_Tasks(self, path, problem):
#         s = 0
#         tasks = pd.DataFrame(columns=['Task', 'Duration'])
#         with open(path+"/"+ problem) as f:
#             contents = f.readlines()
#             for line in contents: 
#                 if line.find(",")==-1:
#                     s = s+1
#                     tasks.loc[len(tasks.index)] = ["T"+str(s),  line.replace("\n","")]
#             f.close()
#         #tasks = tasks.drop([s-1])
#         return tasks;
    
#     def Get_Prec(self, path, problem):
#         Prec2 = pd.DataFrame(columns=['TASK', 'IMMEDIATE_PRECEDESSOR'])
#         with open(path+"/"+ problem) as f:
#             contents = f.readlines()
#             for line in contents: 
#                 if line.find(",")!=-1:
#                     l = line.index(",")
#                     s1 = line[0:l]
#                     s2 = line[l+1:]
#                     s2 = s2.replace("\n","")
#                     if s2 !="-1":
#                         Prec2.loc[len(Prec2.index)] = ["T"+s1,  "T"+s2]
#             f.close()
#         #Prec2.loc[len(Prec2.index)] =["T"+str(len(self.tasks.index)-1), "T"+str(len(self.tasks.index))]
#         for l in self.tasks.index: 
#             t1 = self.tasks.iat[l,0]
#             for p in self.tasks.index: 
#                 if p!= l:
#                     t2 = self.tasks.iat[p,0]
#                     if(len((Prec2.loc[(Prec2['TASK'] == t1)]).loc[(Prec2['IMMEDIATE_PRECEDESSOR']==t2 )])==1):
#                         Prec3 = Prec2[(Prec2['TASK'] == t2)]
#                         for u in range(len(Prec3.index)):
#                             Prec2.loc[len(Prec2.index)] = [t1,  Prec3.iat[u,1]]
#         return Prec2;
    
    def Initialize(self):
        self.tasks = self.Get_Tasks(self.url, self.problem)
        self.tasks["Duration"] = self.tasks["Duration"].astype('float')
        self.Prec = self.Get_Prec(self.url, self.problem)
        self.total_processing_time = self.tasks["Duration"].sum()
        self.N = self.tasks["Duration"].count()
        self.TN_M_WS = math.ceil(self.total_processing_time/self.N)
        self.TN_P_WS = len(self.tasks[self.tasks.Duration >= 2])
        self.WS = max(self.TN_M_WS,self.TN_P_WS) +1 
        self.average_WS_time = self.total_processing_time / self.WS
        #self.Tabu_Structure = self.Create_Tabu_Strcuture(self.tasks, self.Prec)

    
    def get_init_solution(self):
        self.Initialize()
        self.Sol = self.Initial_Solution(len(self.tasks))
        self.Sol = self.Make_Solution_Feasible(self.Sol,self.Prec)
        #Sol = self.Initial_Solution(len(self.tasks))
        #Sol = self.Make_Solution_Feasible(self.Sol,self.Prec)
        return self.Sol
    
    def get_neighbour_solution(self, sol):
        i1 = 0
        i2 = 0
        while i1==i2: 
            i1 = rd.randint(0, len(sol)-1)
            i2 = rd.randint(0, len(sol)-1)
        sol_n = sol[:]
        t1 = sol[i1]
        t2 = sol[i2]
        #sol_n = self.N_Swap(sol, i1, i2)
        for k in self.Tabu_Structure.index:
            if (t1 == self.Tabu_Structure.iloc[k][self.Tabu_Structure.columns[0]] and t2 == self.Tabu_Structure.iloc[k][self.Tabu_Structure.columns[1]] and self.Tabu_Structure.iloc[k][self.Tabu_Structure.columns[2]] =="T") or (t2 == self.Tabu_Structure.iloc[k][self.Tabu_Structure.columns[0]] and t1 == self.Tabu_Structure.iloc[k][self.Tabu_Structure.columns[1]] and self.Tabu_Structure.iloc[k][self.Tabu_Structure.columns[2]] =="T"):
                sol_n = self.N_Swap(sol, i1, i2)
                #self.Tabu_Structure.iat[k, 4] = self.Smoothing_index(sol_n,self.WS,self.tasks)
        return sol_n

    def eval_solution(self, sol):
        SI =self.Smoothing_index(sol, self.WS, self.tasks)
        return SI