import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util.write_file import WriteFile
from util.metrics import calculate_metrics
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def dev_process(opt, model, dev_loader,log_summary_writer = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    y_true, y_pre = [], []
    run_loss, total_labels = 0, 0
    with torch.no_grad():
        for index, (texts, images, labels) in enumerate(tqdm(dev_loader, desc='Dev Iteration', position=0)):
            texts, images, labels = texts, images, labels.to(device)
            
            polarity_logits, intensity_logits, inferred_intensity, direct_intensity_logits, uncertainty= model(texts, images)
            
            loss = nn.CrossEntropyLoss()(polarity_logits, labels)
            loss = loss / opt.acc_batch_size

            _, predicted = torch.max(polarity_logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pre.extend(predicted.cpu().numpy())
            run_loss += loss.item()
            total_labels += labels.size(0)

    run_loss /= total_labels
    dev_metrics = calculate_metrics(y_true, y_pre)
    dev_accuracy, dev_F1_weighted, dev_R_weighted, dev_precision_weighted, dev_F1, dev_R, dev_precision = dev_metrics

    save_content = (f'Dev: Accuracy: {dev_accuracy:.6f}, F1(weighted): {dev_F1_weighted:.6f}, '
                    f'Precision(weighted): {dev_precision_weighted:.6f}, Recall(weighted): {dev_R_weighted:.6f}, '
                    f'F1(macro): {dev_F1:.6f}, Precision: {dev_precision:.6f}, Recall: {dev_R:.6f}, '
                    f'loss: {run_loss:.6f}')
    WriteFile(opt.save_model_path, 'dev_correct_log.txt', save_content + '\n', 'a+')
    print(save_content +'\n')

    if log_summary_writer:
        log_summary_writer.add_scalar('dev_info/acc', dev_accuracy, global_step=1)
        log_summary_writer.add_scalar('dev_info/f1_w', dev_F1_weighted, global_step=1)
        log_summary_writer.add_scalar('dev_info/r_w', dev_R_weighted, global_step=1)
        log_summary_writer.add_scalar('dev_info/p_w', dev_precision_weighted, global_step=1)
        log_summary_writer.add_scalar('dev_info/f1_ma', dev_F1, global_step=1)
        log_summary_writer.flush()

    return dev_accuracy
