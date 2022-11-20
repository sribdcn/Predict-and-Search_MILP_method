import pyscipopt as scp
import argparse
import random
import numpy as np
import os
import torch
from helper import get_a_new2
from pyscipopt import SCIP_PARAMSETTING
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

#4 public datasets, IS, WA, CA, IP
TaskName='IS'
TestNum=100
def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1,delta
    '''
    if task=="IP":
        return 400,5,1
    elif task == "IS":
        return 300,300,15
    elif task == "WA":
        return 0,600,5
    elif task == "CA":
        return 400,0,10
k_0,k_1,delta=test_hyperparam(TaskName)

#set log folder
solver='SCIP'
test_task = f'{TaskName}_{solver}_Predect&Search'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{TaskName}'):
    os.mkdir(f'./logs/{TaskName}')
if not os.path.isdir(f'./logs/{TaskName}/{test_task}'):
    os.mkdir(f'./logs/{TaskName}/{test_task}')
log_folder=f'./logs/{TaskName}/{test_task}'


#load pretrained model
if TaskName=="IP":
    #Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy,postion_get
else:
    from GCN import GNNPolicy
model_name=f'{TaskName}.pth'
pathstr = f'./models/{model_name}'
policy = GNNPolicy().to(DEVICE)
state = torch.load(pathstr, map_location=torch.device('cuda:0'))
policy.load_state_dict(state)


sample_names = sorted(os.listdir(f'./instance/{TaskName}'))
for ins_num in range(TestNum):
    test_ins_name = sample_names[ins_num]
    ins_name_to_read = f'./instance/{TaskName}/{test_ins_name}'

    #get bipartite graph as input
    A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(ins_name_to_read)
    constraint_features = c_nodes.cpu()
    constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
    variable_features = v_nodes
        if TaskName == "IP":
        variable_features = postion_get(variable_features)
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)

    #prediction
    BD = policy(
        constraint_features.to(DEVICE),
        edge_indices.to(DEVICE),
        edge_features.to(DEVICE),
        variable_features.to(DEVICE),
    ).sigmoid().cpu().squeeze()

    #align the variable name betweend the output and the solver
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    scores=[]#get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])


    scores.sort(key=lambda x:x[2],reverse=True)

    scores=[x for x in scores if x[4]=='BINARY']#get binary

    fixer=0
    #fixing variable picked by confidence scores
    count1=0
    for i in range(len(scores)):
        if count1<k_1:
            scores[i][3] = 1
            count1+=1
            fixer += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixer += 1


    print(f'instance: {test_ins_name}, '
          f'fix {k_0} 0s and '
          f'fix {k_1} 1s, delta {delta}. ')

    #read instance
    m1 = scp.Model()
    m1.setParam('limits/time', 1000)
    #m1.hideOutput(True)
    m1.setParam('randomization/randomseedshift', 0)
    m1.setParam('randomization/lpseed', 0)
    m1.setParam('randomization/permutationseed', 0)
    m1.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)#MIP focus
    m1.setLogfile(f'{log_folder}/{test_ins_name}.log')
    m1.readProblem(ins_name_to_read)

    #trust region method implemented by adding constraints
    m1_vars = m1.getVars()
    var_map1 = {}
    for v in m1_vars:  # get a dict (variable map), varname:var clasee
        var_map1[v.name] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = var_map1[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether to fix
        if x_star < 0:
            continue
        tmp_var = m1.addVar(f'alp_{tar_var}_{i}', 'C')
        alphas.append(tmp_var)
        m1.addCons(tmp_var >= tar_var - x_star, f'alpha_up_{i}')
        m1.addCons(tmp_var >= x_star - tar_var, f'alpha_down_{i}')
    m1.addCons(scp.quicksum(ap for ap in alphas) <= delta, 'sum_alpha')
    m1.optimize()







