import os
import argparse
import data_process
import train_process
import torch
import dev_process
import test_process
import model
import random
import numpy as np
from util.write_file import WriteFile
from util.data_paths import prepare_data_paths
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_type', type=int, default=1, help='1: train, 2: test')
    parser.add_argument('-save_model_path', type=str, default='checkpoint', help='Path to save the model')
    parser.add_argument('-epoch', type=int, default=6, help='Number of training epochs')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-acc_grad', type=int, default=1, help='Steps to accumulate gradients')
    parser.add_argument('-bert_lr', type=float, default=4e-5, help='Bert_lr Learning rate')
    parser.add_argument('-lr', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('-image_feature_dim', type=int, default=512, help='Image feature dimension')
    parser.add_argument('-text_feature_dim', type=int, default=512, help='Text feature dimension')
    parser.add_argument('-nhead', type=int, default=8, help='Number of head')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('-num_layers', type=int, default=3, help='Number of Transformer layers')
    parser.add_argument('-seed', type=int, default=42, help='Number of seed')
    parser.add_argument('-num_query_tokens', type=int, default=10, help='Number of opt.num_query_tokens')
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('-dropout2', type=float, default=0.5, help='Dropout2')
    parser.add_argument('-alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('-beta', type=float, default=0.5, help='beta')
    parser.add_argument('-gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('-optim_b1', type=float, default=0.9, help='Adam optimizer beta1')
    parser.add_argument('-optim_b2', type=float, default=0.999, help='Adam optimizer beta2')
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-warmup_ratio', type=float, default=0.01,help='warmup_ratio')
    parser.add_argument('-activate_fun', type=str, default='gelu', help='Activation function')
    parser.add_argument('-add_note', type=str, default='', help='Additional note')
    parser.add_argument('-dataset', type=str, default='MVSA_single', help='Data type (MVSA_single, MVSA_multiple, HFM)')
    parser.add_argument('-save_acc', type=float, default=-1, help='Accuracy threshold for saving the model')
    parser.add_argument('-save_F1', type=float, default=-1, help='F1 score threshold for saving the model')

    return parser.parse_args()


def setup_logging(save_model_path, add_note):
    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if add_note:
        save_model_path += f'/{dt}-{add_note}'
    else:
        save_model_path += f'/{dt}'
    os.makedirs(save_model_path, exist_ok=True)
    log_summary_writer = SummaryWriter(log_dir=save_model_path)
    return log_summary_writer, save_model_path
    

def main():

    opt = parse_args()
    set_seed(opt.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log_summary_writer, opt.save_model_path = setup_logging(opt.save_model_path, opt.add_note)
    
    assert opt.batch_size % opt.acc_grad == 0
    opt.acc_batch_size = opt.batch_size // opt.acc_grad

    senti_model = model.SentiMTLModel(opt).to(device)

    print('Data Ready:')
    data_path_root, photo_path = prepare_data_paths(opt)
    
    train_loader, opt.train_data_len = data_process.data_process(opt, data_path_root['train'], photo_path, data_type=1)
    dev_loader, opt.dev_data_len = data_process.data_process(opt, data_path_root['dev'], photo_path, data_type=2)
    test_loader, opt.test_data_len = data_process.data_process(opt, data_path_root['test'], photo_path, data_type=3)

    log_summary_writer.add_text('Hyperparameter', str(opt), global_step=1)
    log_summary_writer.flush()

    if opt.run_type == 1:
        print('Training Begin:\n')
        # try:
            # train_process.train_process(opt, train_loader, dev_loader, test_loader, senti_model, criterion, log_summary_writer) 
        # except Exception as e:
            # print(f"Training process failed: {e}")
        train_process.train_process(opt, train_loader, dev_loader, test_loader, senti_model, log_summary_writer) 

    elif opt.run_type == 2:
        print('\nTest Begin')
        try:
            model_path = '/checkpoint/bestmodel/bestmodel.pth'
            if os.path.exists(model_path):
                senti_model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=True)
                test_process.test_process(opt, senti_model, test_loader, epoch=1)
            else:
                print(f"Model path {model_path} does not exist!")
        except Exception as e:
            print(f"Test process failed: {e}")
    log_summary_writer.close()

if __name__ == '__main__':

    main()
