import argparse
import os
import time

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.optim import *

from models.dataset import *
from models.modeling_memTxt import *
from utils.utils import check_path, export_config, bool_flag


def eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        for i, data in enumerate(eval_set):
            labels, *input_data = data
            logits, _ = model(*input_data)
            logits = logits.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            logits = logits.argmax(1)
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, logits))
        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
    return accuary, precision, recall, f1, roc_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=15, type=int)
    parser.add_argument('--encoder', default='hita', choices=['hita', 'transformer', 'lsan', 'lstm', 'sand', 'gruself', 'timeline', 'retain', 'retainex'])
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=2, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--max_len', default=50, type=int, help='max visits of EHR')
    parser.add_argument('--max_num_codes', default=20, type=int, help='max number of ICD codes in each visit')
    parser.add_argument('--max_num_blks', default=120, type=int, help='max number of blocks in each visit')
    parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                        help='embedding path of blocks')
    parser.add_argument('--blk_vocab_path', default='./data/processed/block_vocab.txt')
    parser.add_argument('--target_disease', default='Heart failure', choices=['Heart_failure', 'COPD', 'Kidney', 'Dementia', 'Amnesia'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=20, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/', help='model output directory')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)
    

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_auc,test_auc,dev_f1,test_f1\n')

    blk_emb = np.load(args.blk_emb_path)
    blk_pad_id = len(blk_emb) - 1
    if args.target_disease == 'Heart_failure':
        code2id = pickle.load(open('./data/hf/hf_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/hf/hf'
        emb_path = './data/processed/heart_failure.npy'
    elif args.target_disease == 'COPD':
        code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/copd/copd'
        emb_path = './data/processed/COPD.npy'
    elif args.target_disease == 'Kidney':
        code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/kidney/kidney'
        emb_path = './data/processed/kidney_disease.npy'
    elif args.target_disease == 'Dementia':
        code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/dementia/dementia'
        emb_path = './data/processed/dementia.npy'
    elif args.target_disease == 'Amnesia':
        code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/amnesia/amnesia'
        emb_path = './data/processed/amnesia.npy'
    else:
        raise ValueError('Invalid disease')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    train_dataset = MyDataset(data_path + '_training_new.pickle', data_path + '_training_txt.pickle',
                              args.max_len, args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
    dev_dataset = MyDataset(data_path + '_validation_new.pickle', data_path + '_validation_txt.pickle', args.max_len,
                            args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
    test_dataset = MyDataset(data_path + '_testing_new.pickle', data_path + '_testing_txt.pickle', args.max_len,
                             args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.encoder == 'hita':
        model = TextDecoder_hita(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                 args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                                 args.mem_update_size, device)
    elif args.encoder == 'lstm':
        model = TextDecoder_lstm(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                 args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                                 args.mem_update_size, device)
    elif args.encoder == 'lsan':
        model = TextDecoder_lsan(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                 args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                                 args.mem_update_size, device)
    elif args.encoder == 'transformer':
        model = TextDecoder(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                            args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                            args.mem_update_size, device)
    elif args.encoder == 'gruself':
        model = TextDecoder_gruself(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                            args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                            args.mem_update_size, device)
    elif args.encoder == 'timeline':
        model = TextDecoder_timeline(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                            args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                            args.mem_update_size, device)
    elif args.encoder == 'sand':
        model = TextDecoder_sand(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                            args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                            args.mem_update_size, device)
    elif args.encoder == 'retain':
        model = TextDecoder_retain(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                            args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                            args.mem_update_size, device)
    elif args.encoder == 'retainex':
        model = TextDecoder_retainEx(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                            args.max_len, args.blk_emb_path, emb_path, args.target_att_heads, args.mem_size,
                            args.mem_update_size, device)
    else:
        raise ValueError('Invalid encoder')
    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate}
    ]
    optim = Adam(grouped_parameters)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    print('parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\ttotal:', num_params)
    print()
    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_auc, final_test_auc, total_loss = 0.0, 0.0, 0.0
    model.train()
    for epoch_id in range(args.n_epochs):
        print('epoch: {:5} '.format(epoch_id))
        model.train()
        start_time = time.time()
        for i, data in enumerate(train_dataloader):
            labels, *inputs = data
            optim.zero_grad()
            outputs, _ = model(*inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            total_loss += (loss.item() / labels.size(0)) * args.batch_size
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                               total_loss,
                                                                               ms_per_batch))
                total_loss = 0.0
                start_time = time.time()
            global_step += 1

        model.eval()
        train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc = eval_metric(train_dataloader, model)
        dev_acc, d_precision, d_recall, d_f1, d_roc_auc = eval_metric(dev_dataloader, model)
        test_acc, t_precision, t_recall, t_f1, t_roc_auc = eval_metric(test_dataloader, model)
        print('-' * 71)
        print('| step {:5} | train_acc {:7.4f} | dev_acc {:7.4f} | test_acc {:7.4f} '.format(global_step,
                                                                                             train_acc,
                                                                                             dev_acc,
                                                                                             test_acc))
        print(
            '| step {:5} | train_precision {:7.4f} | dev_precision {:7.4f} | test_precision {:7.4f} '.format(
                global_step,
                tr_precision,
                d_precision,
                t_precision))
        print('| step {:5} | train_recall {:7.4f} | dev_recall {:7.4f} | test_recall {:7.4f} '.format(
            global_step,
            tr_recall,
            d_recall,
            t_recall))
        print('| step {:5} | train_f1 {:7.4f} | dev_f1 {:7.4f} | test_f1 {:7.4f} '.format(global_step,
                                                                                          tr_f1,
                                                                                          d_f1,
                                                                                          t_f1))
        print('| step {:5} | train_auc {:7.4f} | dev_auc {:7.4f} | test_auc {:7.4f} '.format(global_step,
                                                                                             tr_roc_auc,
                                                                                             d_roc_auc,
                                                                                             t_roc_auc))
        print('-' * 71)

        if d_roc_auc >= best_dev_auc:
            best_dev_auc = d_roc_auc
            final_test_auc = t_roc_auc
            best_dev_epoch = epoch_id
            torch.save([model, args], model_path)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{},{},{}\n'.format(global_step, d_roc_auc, t_roc_auc, d_f1, t_f1))
            print(f'model saved to {model_path}')
        if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
    print('final test auc: {:.4f}'.format(final_test_auc))
    print()


def eval(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.target_disease_vectors.to(device)
    model.eval()

    blk_emb = np.load(old_args.blk_emb_path)
    blk_pad_id = len(blk_emb) - 1
    if old_args.target_disease == 'Heart_failure':
        code2id = pickle.load(open('./data/hf/hf_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/hf/hf_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/hf/hf'
    elif old_args.target_disease == 'COPD':
        code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/copd/copd_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/copd/copd'
    elif old_args.target_disease == 'Kidney':
        code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/kidney/kidney_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/kidney/kidney'
    elif old_args.target_disease == 'Amnesia':
        code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/amnesia/amnesia_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/amnesia/amnesia'
    elif old_args.target_disease == 'Dementia':
        code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/dementia/dementia_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/dementia/dementia'
    else:
        raise ValueError('Invalid disease')

    dev_dataset = MyDataset(data_path + '_validation_new.pickle', data_path + '_validation_txt.pickle',
                            old_args.max_len, old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
    test_dataset = MyDataset(data_path + '_testing_new.pickle', data_path + '_testing_txt.pickle', old_args.max_len,
                             old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    with open(args.blk_vocab_path, 'r', encoding='utf-8') as fin:
        vocab = [line.strip() for line in fin]
    icd2des = {}
    df = pd.read_excel('./data/CMS32_DESC_LONG_SHORT_DX.xlsx')
    icds = df['DIAGNOSIS CODE']
    desc = df['LONG DESCRIPTION']
    for i in range(len(icds)):
        icd = icds[i][0:3] + '.' + icds[i][3:]
        icd2des[icd] = desc[i]
    with open(os.path.join(args.save_dir, 'decode_test.txt'), 'w') as fout:
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            labels, ehr, *inputs = data
            logits, memory = model(ehr, *inputs)
            predictions = logits.argmax(1)
            for l, pred, record, mem in zip(labels, predictions, ehr, memory):
                output = {}
                output["label"] = l.item()
                output["prediction"] = pred.item()
                record = record.data.cpu().numpy()
                # icd
                rec_li = []
                for visit in record:
                    v_li = []
                    for codeid in visit:
                        if codeid in id2code.keys():
                            v_li.append(id2code[codeid])
                    if len(v_li) > 0:
                        rec_li.append(v_li)
                    else:
                        break
                output["record_icd"] = rec_li
                # icd text
                des_li = []
                for vi in rec_li:
                    v_des = []
                    for code in vi:
                        if code in icd2des.keys():
                            v_des.append(icd2des[code])
                    des_li.append(v_des)
                output["record_text"] = des_li
                # memory
                num_vi = len(rec_li)
                mem = mem.data.cpu().numpy()[0: num_vi - 1]
                mem_txt = []
                for v_mem in mem:
                    v_txt = []
                    for txt_id in v_mem:
                        if txt_id < len(vocab):
                            v_txt.append(vocab[txt_id])
                    mem_txt.append(v_txt)
                output["memory"] = mem_txt
                fout.write(str(output))
                fout.write('\n\n')


if __name__ == '__main__':
    main()
