import os.path
import pickle
from multiprocessing import Process, Queue
import gurobipy as gp
import numpy as np
import argparse
from helper import get_a_new2

def solve_grb(filepath,log_dir,settings):
    gp.setParam('LogToConsole', 0)
    m = gp.read(filepath)

    m.Params.PoolSolutions = settings['maxsol']
    m.Params.PoolSearchMode = settings['mode']
    m.Params.TimeLimit = settings['maxtime']
    m.Params.Threads = settings['threads']
    log_path = os.path.join(log_dir, os.path.basename(filepath)+'.log')
    with open(log_path,'w'):
        pass

    m.Params.LogFile = log_path
    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr('SolCount')

    mvars = m.getVars()
    #get variable name,
    oriVarNames = [var.varName for var in mvars]

    varInds=np.arange(0,len(oriVarNames))

    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)


    sols = np.array(sols,dtype=np.float32)
    objs = np.array(objs,dtype=np.float32)

    sol_data = {
        'var_names': oriVarNames,
        'sols': sols,
        'objs': objs,
    }

    return sol_data

def collect(ins_dir,q,sol_dir,log_dir,bg_dir,settings):

    while True:
        filename = q.get()
        if not filename:
            break
        filepath = os.path.join(ins_dir,filename)        
        sol_data = solve_grb(filepath,log_dir,settings)
        #get bipartite graph , binary variables' indices
        A2,v_map2,v_nodes2,c_nodes2,b_vars2=get_a_new2(filepath)
        BG_data=[A2,v_map2,v_nodes2,c_nodes2,b_vars2]
        
        # save data
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
        pickle.dump(BG_data, open(os.path.join(bg_dir, filename+'.bg'), 'wb'))





if __name__ == '__main__':
    #sizes=['small','large']
    sizes=["IP","WA","IS","CA","NNV"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--nWorkers', type=int, default=100)
    parser.add_argument('--maxTime', type=int, default=3600)
    parser.add_argument('--maxStoredSol', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()


    for size in sizes:
    

        dataDir = args.dataDir

        INS_DIR = os.path.join(dataDir,f'instance/train/{size}')

        if not os.path.isdir(f'./dataset/{size}'):
            os.mkdir(f'./dataset/{size}')
        if not os.path.isdir(f'./dataset/{size}/solution'):
            os.mkdir(f'./dataset/{size}/solution')
        if not os.path.isdir(f'./dataset/{size}/NBP'):
            os.mkdir(f'./dataset/{size}/NBP')
        if not os.path.isdir(f'./dataset/{size}/logs'):
            os.mkdir(f'./dataset/{size}/logs')
        if not os.path.isdir(f'./dataset/{size}/BG'):
            os.mkdir(f'./dataset/{size}/BG')

        SOL_DIR =f'./dataset/{size}/solution'
        LOG_DIR =f'./dataset/{size}/logs'
        BG_DIR =f'./dataset/{size}/BG'
        os.makedirs(SOL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        os.makedirs(BG_DIR, exist_ok=True)

        N_WORKERS = args.nWorkers

        # gurobi settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'mode': 2,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,

        }

        filenames = os.listdir(INS_DIR)
   
        q = Queue()
        # add ins
        for filename in filenames:
            if not os.path.exists(os.path.join(BG_DIR,filename+'.bg')):
                q.put(filename)
        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect,args=(INS_DIR,q,SOL_DIR,LOG_DIR,BG_DIR,SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        print('done')


