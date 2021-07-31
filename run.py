import argparse
import os
import glob
import time
import numpy as np
import torch
import torch.optim as optim
 
from dataset.scannetv2 import Dataset
from network.VMNet import VMNet, model_fn

from utils import PolyLR
from utils import evaluate


def train(exp_name, train_files, MAX_ITER = 80000):
    
    # Saving directory
    directory = 'check_points/' + exp_name
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Model & model_fn
    model = VMNet()
    model = model.cuda()
    m_fn = model_fn(inference = False)
    
    # Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr = 1e-1, momentum = 0.9, dampening = 0.1, weight_decay = 1e-4)
    scheduler = PolyLR(optimizer, max_iter = MAX_ITER, power = 0.9, last_step = -1)

    # Load training statics
    curr_iter = 0
    PATH = directory + '/current.pt'
    if os.path.exists(PATH):    
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_iter = checkpoint['iteration']
        scheduler = PolyLR(optimizer, max_iter = MAX_ITER, power = 0.9, last_step = curr_iter)
        print("Restored from: {}".format(PATH))
    

    # Training process
    is_training = True
    
    # DataLoader
    dataset = Dataset(train_files = train_files)
    train_data_loader = dataset.train_data_loader()

    while is_training:
        
        model.train()
        
        for i, batch in enumerate(train_data_loader):
            
            print("Iteration: {}, lr: {}".format(curr_iter, optimizer.param_groups[0]['lr']))
            
            optimizer.zero_grad()

            loss = m_fn(model, batch)

            loss.backward()

            # Update    
            optimizer.step()
            scheduler.step()

            print('loss: {}'.format(loss.item()))

            if curr_iter >= MAX_ITER:
                is_training = False
                break

            curr_iter += 1

            if curr_iter % 500 == 0:
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': curr_iter,
                        }, directory + '/current.pt')    


def val(exp_name, val_files, val_reps = 8):

    # Model & model_fn
    model = VMNet()
    model = model.cuda()
    m_fn = model_fn(inference = True)
    print('#parameters', sum([x.nelement() for x in model.parameters()]))

    # Saving directory
    directory = 'check_points/' + exp_name
    PATH = directory + '/current.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Restored from: {}".format(PATH))

    # DataLoader
    dataset = Dataset(infer_files = val_files, val = True)
    infer_data_loader = dataset.infer_data_loader()

    # Evaluation process
    with torch.no_grad():            

        model.eval()

        store = torch.zeros(dataset.infer_offsets[-1], 20)
        
        start = time.time()

        print("*****************************Validation*************************************")

        for rep in range(1, 1 + val_reps):
            
            for i, batch in enumerate(infer_data_loader):
                
                predictions = m_fn(model, batch) 
                store.index_add_(0, batch['point_ids'], predictions)
            
            print('rep ', rep, ' time=', time.time() - start, 's')
            evaluate(store.max(1)[1].numpy(), dataset.infer_labels)
        


def test(exp_name, test_files, test_reps = 8):
    
    # Model & model_fn
    model = VMNet()
    model = model.cuda()
    m_fn = model_fn(inference = True)
    print('#parameters', sum([x.nelement() for x in model.parameters()]))

    # Saving directory
    directory = 'check_points/' + exp_name
    PATH = directory + '/current.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Restored from: {}".format(PATH))    

    # DataLoader
    dataset = Dataset(infer_files = test_files)
    infer_data_loader = dataset.infer_data_loader()

    # Evaluation process
    with torch.no_grad():            
        
        model.eval()

        store = torch.zeros(dataset.infer_offsets[-1], 20)
        
        start = time.time()

        print("*****************************Test*************************************")

        for rep in range(1, 1 + test_reps):
            
            for i, batch in enumerate(infer_data_loader):
                
                predictions = m_fn(model, batch) 
                store.index_add_(0, batch['point_ids'], predictions)
            
            print('rep ', rep, ' time=', time.time() - start, 's')

    inverse_mapper = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])

    predictions = store.max(1)[1].numpy()

    for idx, test_file in enumerate(test_files):

        pred = predictions[dataset.infer_offsets[idx] : dataset.infer_offsets[idx + 1]]

        ori_pred = np.array([inverse_mapper[i] for i in pred])
        ori_pred = ori_pred.astype(np.int32)

        test_name = 'test_results/' + test_file[-15:-3] + '.txt'

        np.savetxt(test_name, ori_pred, fmt='%d', encoding='utf8')
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'VMNet Processing')
    parser.add_argument('--train', dest = 'train', action = 'store_true')
    parser.set_defaults(train = False)
    parser.add_argument('--train_benchmark', dest = 'train_b', action = 'store_true')
    parser.set_defaults(train_b = False)
    parser.add_argument('--val', dest = 'val', action = 'store_true')
    parser.set_defaults(val = False)
    parser.add_argument('--test', dest = 'test', action = 'store_true')
    parser.set_defaults(test = False)
    parser.add_argument('--exp_name', default = 'VMNet', type = str, required = True,
                        help = 'name of the experiment')
    parser.add_argument('--data_path', default = None, type = str, required = True,
                        help = 'path to preprocessed data, should have subfolders train/ val/ test/')
    args = parser.parse_args()
    
    # Take one process a time
    assert np.sum([args.train, args.train_b, args.val, args.test]) == 1, 'Please select one and only one task'
    
    # Check cuda
    use_cuda = torch.cuda.is_available()
    print("use_cuda: {}".format(use_cuda))
    if use_cuda is False:
        raise ValueError("CUDA is not available!")
    
    # Check exp
    exp_name = args.exp_name
    print("exp_name: {}".format(exp_name))

    # Data paths
    train_files_path = args.data_path + '/train'
    val_files_path = args.data_path + '/val'
    test_files_path = args.data_path + '/test'
    train_files = sorted(glob.glob(train_files_path + '/*.pt'))
    val_files = sorted(glob.glob(val_files_path + '/*.pt'))
    test_files = sorted(glob.glob(test_files_path + '/*.pt'))

    if args.train:
        train(exp_name, train_files)
    if args.train_b:
        train(exp_name, train_files + val_files)
    if args.val:
        val(exp_name, val_files)
    if args.test:
        test(exp_name, test_files)
