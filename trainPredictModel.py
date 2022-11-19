import os
import torch
import torch_geometric
import random
import time
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#this file is to train a predict model. given a instance's bipartite graph as input, the model predict the binary distribution.

#4 public datasets, IS, WA, CA, IP
TaskName="WA"

#set folder
train_task=f'{TaskName}_train'
if not os.path.isdir(f'./train_logs'):
    os.mkdir(f'./train_logs')
if not os.path.isdir(f'./train_logs/{train_task}'):
    os.mkdir(f'./train_logs/{train_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain')
if not os.path.isdir(f'./pretrain/{train_task}'):
    os.mkdir(f'./pretrain/{train_task}')
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}_train.log', 'wb')

#set params
LEARNING_RATE = 0.001
NB_EPOCHS =9999
BATCH_SIZE = 4
NUM_WORKERS = 0
WEIGHT_NORM=100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR_BG = f'./dataset/{TaskName}/BG'
DIR_SOL = f'./dataset/{TaskName}/solution'
sample_names = os.listdir(DIR_BG)
sample_files = [ (os.path.join(DIR_BG,name), os.path.join(DIR_SOL,name).replace('bg','sol')) for name in sample_names]
random.seed(0)
random.shuffle(sample_files)
train_files = sample_files[: int(0.80 * len(sample_files))]
valid_files = sample_files[int(0.80 * len(sample_files)) :]
if TaskName=="IP":
    #Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy
    from GCN import GraphDataset_position as GraphDataset
else:
    from GCN import GraphDataset, GNNPolicy

train_data = GraphDataset(train_files)
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

PredictModel = GNNPolicy().to(DEVICE)

def EnergyWeightNorm(task):
    if task=="IP":
        return 1
    elif task=="WA":
        return 100
    elif task == "IS":
        return -100
    elif task == "CA":
        return -1000

def train(predict, data_loader, optimizer=None,weight_norm=1):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        predict.train()
    else:
        predict.eval()
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)
            # get target solutions in list format
            solInd = batch.nsols
            target_sols = []
            target_vals = []
            solEndInd = 0
            valEndInd = 0

            for i in range(solInd.shape[0]):#for in batch
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)

            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 #remove nan value
            #predict the binary distribution, BD
            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            BD = BD.sigmoid()
    
            # compute loss
            loss = 0
            # calculate weights
            index_arrow = 0
            # print("start calculate loss  :")
            for ind,(sols,vals) in enumerate(zip(target_sols,target_vals)):

                #compute weight
                n_vals = vals
                exp_weight = torch.exp(-n_vals/weight_norm)
                weight = exp_weight/exp_weight.sum()

                # get a binary mask
                varInds = batch.varInds[ind]
                varname_map=varInds[0][0]
                b_vars=varInds[1][0].long()

                #get binary variables
                sols = sols[:,varname_map][:,b_vars]

                # cross-entropy
                n_var = batch.ntvars[ind]
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                index_arrow = index_arrow + n_var
                pos_loss = -(pre_sols+ 1e-8).log()[None,:]*(sols==1).float()
                neg_loss = -(1-pre_sols + 1e-8).log()[None,:]*(sols==0).float()
                sum_loss = pos_loss + neg_loss

                sample_loss = sum_loss*weight[:,None]
                loss += sample_loss.sum()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed

    return mean_loss

optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)



weight_norm=EnergyWeightNorm(TaskName)
best_val_loss = 99999
for epoch in range(NB_EPOCHS):

    begin=time.time()
    train_loss = train(PredictModel, train_loader, optimizer,weight_norm)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
    valid_loss = train(PredictModel, valid_loader, None,weight_norm)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
    if valid_loss<best_val_loss:
        best_val_loss = valid_loss
        torch.save(PredictModel.state_dict(),model_save_path+'model_best.pth')
    torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
    st = f'@epoch{epoch}   Train loss:{train_loss}   Valid loss:{valid_loss}    TIME:{time.time()-begin}\n'
    log_file.write(st.encode())
    log_file.flush()
print('done')





