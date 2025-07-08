import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from util.write_file import WriteFile
from util.metrics import calculate_metrics
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_process(opt, model, test_loader, log_summary_writer = None, epoch=None):
    y_true, y_pre = [], []
    test_loss, total_labels = 0, 0

    with torch.no_grad():
        model.eval()
        test_loader_tqdm = tqdm(test_loader, desc='Test Iteration', position=0)
        #******可视化*******
        features = []
        label_list = []
        for texts, images, labels in test_loader_tqdm:
            texts, images, labels = texts, images, labels.to(device)
            
            polarity_logits, intensity_logits, inferred_intensity, direct_intensity_logits, uncertainty = model(texts, images)

            loss = nn.CrossEntropyLoss()(polarity_logits, labels) 
            loss = loss / opt.acc_batch_size
            test_loss += loss.item()

            _, predicted = torch.max(polarity_logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pre.extend(predicted.cpu().numpy())
            total_labels += labels.size(0)

            test_loader_tqdm.set_description(f"Test Iteration, loss: {loss:.6f}")

            label_list.append(labels.cpu().numpy())

        test_loss /= total_labels
        test_metrics = calculate_metrics(np.array(y_true), np.array(y_pre))
        test_accuracy, test_F1_weighted, test_R_weighted, test_precision_weighted, test_F1, test_R, test_precision = test_metrics

        save_content = (f'Test : Accuracy: {test_accuracy:.6f}, F1(weighted): {test_F1_weighted:.6f}, '
                        f'Precision(weighted): {test_precision_weighted:.6f}, Recall(weighted): {test_R_weighted:.6f}, '
                        f'F1(macro): {test_F1:.6f}, Precision: {test_precision:.6f}, Recall: {test_R:.6f}, '
                        f'loss: {test_loss:.6f}')

        WriteFile(opt.save_model_path, 'test_correct_log.txt', save_content + '\n', 'a+')
        print(save_content)
        print('\n')
        if log_summary_writer:
            log_summary_writer.add_scalar('test_info/loss_epoch', test_loss, global_step=epoch)
            log_summary_writer.add_scalar('test_info/acc', test_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_w', test_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/r_w', test_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/p_w', test_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_ma', test_F1, global_step=epoch)
            log_summary_writer.flush()

