from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data import generator
from utils import io_utils
import models.models as models
import test
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
parser.add_argument('--exp_name', type=str, default='debug_vx', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--iterations', type=int, default=50000, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
                    help='Learning rate decay interval')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=300000, metavar='N',
                    help='how many batches between each model saving')
parser.add_argument('--test_interval', type=int, default=2000, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--test_N_way', type=int, default=5, metavar='N',
                    help='Number of classes for doing each classification run')
parser.add_argument('--train_N_way', type=int, default=5, metavar='N',
                    help='Number of classes for doing each training comparison')
parser.add_argument('--test_N_shots', type=int, default=1, metavar='N',
                    help='Number of shots in test')
parser.add_argument('--train_N_shots', type=int, default=1, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--metric_network', type=str, default='gnn_iclr_nl', metavar='N',
                    help='gnn_iclr_nl' + 'gnn_iclr_active')
parser.add_argument('--active_random', type=int, default=0, metavar='N',
                    help='random active ? ')
parser.add_argument('--dataset_root', type=str, default='datasets', metavar='N',
                    help='Root dataset')
parser.add_argument('--test_samples', type=int, default=30000, metavar='N',
                    help='Number of shots')
parser.add_argument('--dataset', type=str, default='mini_imagenet', metavar='N',
                    help='omniglot')
parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                    help='Decreasing the learning rate every x iterations')
args = parser.parse_args()


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp models/models.py checkpoints' + '/' + args.exp_name + '/' + 'models.py.backup')
_init_()

io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
io.cprint(str(args))

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    io.cprint('Using GPU : ' + str(torch.cuda.current_device())+' from '+str(torch.cuda.device_count())+' devices')
    torch.cuda.manual_seed(args.seed)
else:
    io.cprint('Using CPU')


def train_batch(model, data):
    [enc_nn, metric_nn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi, hidden_labels] = data

    # Compute embedding from x and xi_s
    z = enc_nn(batch_x)[-1]
    zi_s = [enc_nn(batch_xi)[-1] for batch_xi in batches_xi]

    # Compute metric from embeddings
    out_metric, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
    logsoft_prob = softmax_module.forward(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()

    return loss


def train():
    train_loader = generator.Generator(args.dataset_root, args, partition='train', dataset=args.dataset)
    io.cprint('Batch size: '+str(args.batch_size))

    #Try to load models
    enc_nn = models.load_model('enc_nn', args, io)
    metric_nn = models.load_model('metric_nn', args, io)

    if enc_nn is None or metric_nn is None:
        enc_nn, metric_nn = models.create_models(args=args)
    softmax_module = models.SoftmaxModule()

    if args.cuda:
        enc_nn.cuda()
        metric_nn.cuda()

    io.cprint(str(enc_nn))
    io.cprint(str(metric_nn))

    weight_decay = 0
    if args.dataset == 'mini_imagenet':
        print('Weight decay '+str(1e-6))
        weight_decay = 1e-6
    opt_enc_nn = optim.Adam(enc_nn.parameters(), lr=args.lr, weight_decay=weight_decay)
    opt_metric_nn = optim.Adam(metric_nn.parameters(), lr=args.lr, weight_decay=weight_decay)

    enc_nn.train()
    metric_nn.train()
    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    test_acc = 0
    for batch_idx in range(args.iterations):

        ####################
        # Train
        ####################
        data = train_loader.get_task_batch(batch_size=args.batch_size, n_way=args.train_N_way,
                                           unlabeled_extra=args.unlabeled_extra, num_shots=args.train_N_shots,
                                           cuda=args.cuda, variable=True)
        [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi, hidden_labels] = data

        opt_enc_nn.zero_grad()
        opt_metric_nn.zero_grad()

        loss_d_metric = train_batch(model=[enc_nn, metric_nn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi, hidden_labels])

        opt_enc_nn.step()
        opt_metric_nn.step()


        adjust_learning_rate(optimizers=[opt_enc_nn, opt_metric_nn], lr=args.lr, iter=batch_idx)

        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss_d_metric.data[0]
        if batch_idx % args.log_interval == 0:
                display_str = 'Train Iter: {}'.format(batch_idx)
                display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss/counter)
                io.cprint(display_str)
                counter = 0
                total_loss = 0

        ####################
        # Test
        ####################
        if (batch_idx + 1) % args.test_interval == 0 or batch_idx == 20:
            if batch_idx == 20:
                test_samples = 100
            else:
                test_samples = 3000
            if args.dataset == 'mini_imagenet':
                val_acc_aux = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                                 test_samples=test_samples*5, partition='val')
            test_acc_aux = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                              test_samples=test_samples*5, partition='test')
            test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                               test_samples=test_samples, partition='train')
            enc_nn.train()
            metric_nn.train()

            if val_acc_aux is not None and val_acc_aux >= val_acc:
                test_acc = test_acc_aux
                val_acc = val_acc_aux

            if args.dataset == 'mini_imagenet':
                io.cprint("Best test accuracy {:.4f} \n".format(test_acc))

        ####################
        # Save model
        ####################
        if (batch_idx + 1) % args.save_interval == 0:
            torch.save(enc_nn, 'checkpoints/%s/models/enc_nn.t7' % args.exp_name)
            torch.save(metric_nn, 'checkpoints/%s/models/metric_nn.t7' % args.exp_name)

    # Test after training
    test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                       test_samples=args.test_samples)


def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5**(int(iter/args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == "__main__":
    train()


