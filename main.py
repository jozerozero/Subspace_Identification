from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.logger import CompleteLogger
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy
from common.utils.data import ForeverDataIterator
from common.modules.my_networks import iVAE
import utils
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
from solver import BBSL
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
sys.path.append('.')

os.environ["WANDB_MODE"] = "disabled"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    cudnn.benchmark = True
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, args.num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source,
                          args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=(args.n_domains-1)*args.train_batch_size,
                                     num_workers=args.workers, drop_last=True,
                                     # sampler=_make_balanced_sampler(train_source_dataset.domain_ids)
                                     shuffle=True,
                                     )
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True,
                                     )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    classifier = iVAE(args, backbone_net=backbone).to(device)

    optimizer = SGD(classifier.get_parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    print(optimizer.param_groups[0]['lr'], ' *** lr')
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(optimizer.param_groups[0]['lr'], ' *** lr')
    test_logger = '%s/test.txt' % (args.log)

    sample_weight = torch.Tensor(
        np.ones([args.n_domains-1, args.num_classes])).to(device)

    best_acc1 = 0.0
    total_iter = 0.0
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr(),
              optimizer.param_groups[0]['lr'])

        my_confuse_matrix, tgt_pseudo_label, src_true = train(train_source_iter=train_source_iter,
                                                              train_target_iter=train_target_iter, model=classifier,
                                                              optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                              epoch=epoch, args=args, total_iter=total_iter, backbone=backbone,
                                                              sample_weight=sample_weight)
        if args.is_use_tgt_shift:
            sample_weight = conv_optimize(
                args, my_confuse_matrix, src_true, tgt_pseudo_label, sample_weight)

        total_iter += args.iters_per_epoch
        acc1 = utils.validate(val_loader=val_loader,
                              model=classifier, args=args, device=device)
        print(' * Val Acc@1 %.3f' % (acc1))
        wandb.log({"Val Acc": acc1})
        if args.data.lower() == "domainnet":
            acc1 = utils.validate(test_loader, classifier, args, device)
        wandb.log({"Test Acc": acc1})
        message = '(epoch %d): Test Acc@1 %.3f' % (epoch+1, acc1)
        print(message)
        record = open(test_logger, 'a')
        record.write(message+'\n')
        record.close()

        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        wandb.run.summary["best_accuracy"] = best_acc1

    print("best_acc1 = {:3.1f}".format(best_acc1))
    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("Final Best test_acc1 = {:3.1f}".format(acc1))
    logger.close()


def conv_optimize(args, my_confuse_matrix, src_true, tgt_pseudo_label, sample_weight):

    src_true = src_true.cpu()
    tgt_pseudo_label = tgt_pseudo_label / np.sum(tgt_pseudo_label)

    for domain_idx in range(args.n_domains-1):
        Con_s = my_confuse_matrix[:, :, domain_idx]
        Con_s = Con_s / np.sum(Con_s)
        src_true_s = src_true[domain_idx, :]
        result = BBSL(C=Con_s, y_t=tgt_pseudo_label, y_s=src_true_s)
        sample_weight[domain_idx, :] = torch.tensor(
            result, requires_grad=False).to(device)

    return sample_weight


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: iVAE, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, total_iter, backbone, sample_weight):

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    recon_losses = AverageMeter('Rec', ':4.2f')
    vae_losses = AverageMeter('VAE', ':4.2f')
    kl_losses = AverageMeter('KL', ':4.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')
    ent_losses = AverageMeter('Ent', ':4.2f')
    sem_losses = AverageMeter('Sem', ":4.2f")
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    val_accs = AverageMeter('Val Acc', ':3.1f')
    dom_accs = AverageMeter('Dom Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, ent_losses, vae_losses,
            recon_losses, kl_losses, cls_accs, val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )
    normal_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(args.z_dim).cuda(), torch.eye(args.z_dim).cuda())

    model.train()

    my_confuse_matrix = np.zeros(
        [args.num_classes, args.num_classes, args.n_domains-1])
    tgt_pseudo_label = np.zeros([args.num_classes])

    src_true = torch.Tensor(
        np.zeros([args.n_domains - 1, args.num_classes])).to(device)
    for domain_idx in range(args.n_domains-1):
        src_labels = np.array(
            [y for x, y in train_source_iter.data_loader.dataset.datasets[domain_idx].samples])
        for cls_idx in range(args.num_classes):
            src_true[domain_idx, cls_idx] = np.count_nonzero(
                src_labels == cls_idx)
        src_true[domain_idx, :] = src_true[domain_idx, :] / len(src_labels)

    domain_avg_weight = torch.Tensor(
        np.ones([args.n_domains-1]) / (args.n_domains-1))
    # tgt_estimated_distri = torch.dot(
    #     domain_avg_weight, torch.multiply(sample_weight, src_true))

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()

        data_time.update(time.time() - end)

        img_s, labels_s, d_s, _ = next(train_source_iter)
        img_t, labels_t, d_t, _ = next(train_target_iter)
        img_s = img_s.to(device)
        img_t = img_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        img_all = torch.cat([img_s, img_t], 0)
        d_all = torch.cat([d_s, d_t], 0).to(device)
        label_all = torch.cat([labels_s, labels_t], 0)

        losses_cls = []
        losses_domain = []
        losses_kl = []
        z_all = []
        y_t = None
        y_s = []
        labels_s = []
        x_all = []

        src_feat_list = list()
        tgt_feat = None

        batch_confuse_matrix = np.zeros(
            [args.num_classes, args.num_classes, args.n_domains-1])
        batch_tgt_pseudo_label = np.zeros([args.num_classes])

        for id in range(args.n_domains):
            domain_id = id
            is_target = domain_id == args.n_domains-1

            index = d_all == id
            label_dom = label_all[index] if not is_target else None
            img_dom = img_all[index]
            d_dom = d_all[index]
            x_dom = model.backbone(img_dom, track_bn=is_target)
            z, z_s1, z_s2, z_c1, z_c2, label_logit, domain_logit, mu, log_var = model.encode(
                x=x_dom, u=d_dom, track_bn=is_target)

            if is_target:
                tgt_feat = z_c1
            else:
                src_feat_list.append(z_c1)

            q_dist = torch.distributions.Normal(
                mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
            log_qz = q_dist.log_prob(z)
            log_pz = normal_distribution.log_prob(z)
            kl = (log_qz.sum(dim=1) - log_pz).mean()
            C = torch.clamp(torch.tensor(args.C_max) /
                            args.C_stop_iter * total_iter, 0, args.C_max)
            loss_kl = args.beta * (kl - C).abs()

            if not is_target:
                losses_cls.append(F.cross_entropy(
                    label_logit, label_dom, weight=sample_weight[id, :]))
                y_s.append(label_logit)
                labels_s.append(label_dom)

                with torch.no_grad():
                    one_src_pred = torch.argmax(
                        label_logit, dim=1).cpu().numpy()
                    one_src_label = label_dom.cpu().numpy()
                    batch_confuse_matrix[:, :, id] = confusion_matrix(one_src_label, one_src_pred,
                                                                      labels=list(range(args.num_classes)))
                    my_confuse_matrix += batch_confuse_matrix
            else:
                outputs_target = y_t = label_logit

                # mcc loss
                train_bs = img_t.shape[0]

                outputs_target_temp = outputs_target / args.temperature
                target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
                target_entropy_weight = Entropy(target_softmax_out_temp).detach()
                target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
                target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
                cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
                    target_softmax_out_temp)
                cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
                mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / args.class_num


                tgt_pred = torch.argmax(y_t, dim=1).cpu().numpy()
                for cls_idx in range(args.num_classes):
                    batch_tgt_pseudo_label[cls_idx] = np.count_nonzero(
                        tgt_pred == cls_idx)

                tgt_pseudo_label += batch_tgt_pseudo_label

            losses_kl.append(loss_kl)
            losses_domain.append(F.cross_entropy(domain_logit, d_dom))
            x_all.append(x_dom)
            z_all.append(z)

        if args.is_label_conditional_alignment:
            # print("use is_label_conditional_alignment")
            semantic_loss_list = []
            for domain_idx in range(args.n_domains-1):
                tgt_label_estimated_distri = sample_weight[domain_idx,
                                                        :] * src_true[domain_idx, :]
                semantic_loss = model.update_center2(
                    domain_idx=domain_idx, src_feature=src_feat_list[domain_idx], tgt_feature=tgt_feat, 
                    src_truth_label=labels_s[domain_idx], tgt_pseudo_label=torch.tensor(tgt_pred).to(device), tgt_y_estimated=tgt_label_estimated_distri)
                semantic_loss_list.append(semantic_loss)
            total_semantic_loss = torch.mean(torch.stack(semantic_loss_list))
            # print(semantic_loss_list)
            # print(total_semantic_loss)
            # exit()
        else:
            total_semantic_loss = 0.0

        x_all = torch.cat(x_all, 0)
        z_all = torch.cat(z_all, 0)
        x_all_hat = model.decode(z_all)

        mean_loss_recon = F.mse_loss(
            x_all, x_all_hat, reduction="sum") / len(x_all)
        mean_loss_kl = torch.stack(losses_kl, dim=0).mean()
        mean_loss_vae = mean_loss_recon + mean_loss_kl

        mean_loss_cls = torch.stack(losses_cls, 0).mean()
        mean_loss_domain = torch.stack(losses_domain, 0).mean()

        # entropy loss
        loss_ent = torch.tensor(0.0).to(device)
        if args.lambda_ent > 0:
            output_t = y_t
            entropy = F.cross_entropy(output_t, torch.softmax(
                output_t, dim=1), reduction="none").detach()
            index = torch.nonzero(
                (entropy < args.entropy_thr).float()).squeeze(-1)
            select_output_t = output_t[index]
            if len(select_output_t) > 0:
                loss_ent = F.cross_entropy(
                    select_output_t, torch.softmax(select_output_t, dim=1))

        loss = mean_loss_cls \
            + args.lambda_dom * mean_loss_domain\
            + args.lambda_vae * mean_loss_vae \
            # + args.lambda_ent * loss_ent * 1 \
            + args.lambda_sem * total_semantic_loss * 1 + args.mcc_weight * mcc_loss

        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]
        cls_losses.update(mean_loss_cls.item(), y_s.size(0))
        recon_losses.update(mean_loss_recon.item(), x_all.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))
        vae_losses.update(mean_loss_vae.item(), x_all.size(0))
        ent_losses.update(loss_ent.item(), y_t.size(0))
        kl_losses.update(mean_loss_kl.item(), x_all.size(0))
        sem_losses.update(total_semantic_loss, y_s.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            model.eval()
            img_t = img_all[d_all == args.n_domains-1]
            labels_t = label_all[d_all == args.n_domains-1]
            with torch.no_grad():
                y, d = model(img_t, d_all[d_all == args.n_domains-1])
                cls_t_acc = accuracy(y, labels_t)[0]
                cls_d_acc = accuracy(d, d_all[d_all == args.n_domains-1])[0]
                val_accs.update(cls_t_acc.item(), img_t.size(0))
                dom_accs.update(cls_d_acc.item(), img_t.size(0))
            model.train()

            progress.display(i)

            wandb.log({
                "Train Target Acc": cls_t_acc.item(),
                "Train Source Acc": cls_acc.item(),
                "Train Source Cls Loss": mean_loss_cls.item(),
                "Train Reconstruction Loss": mean_loss_recon.item(),
                "Train VAE Loss": mean_loss_vae.item(),
                "Entropy Loss": loss_ent.item(),
                "Train KL": mean_loss_kl.item(),
            })

    return my_confuse_matrix, tgt_pseudo_label, src_true


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


if __name__ == "__main__":
    base_path = "/l/users/$user_name$/da_datasets/"

    parser = argparse.ArgumentParser(description='Subspace Domain Adaptation')

    parser.add_argument("--root", type=str,
                        default=os.path.join(base_path, "office-home"),
                        help="root path of dataset")
    parser.add_argument("--data", default="OfficeHome", help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                        ' (default: Office31)')
    parser.add_argument("-s", "--source", default="Cl,Pr,Rw")
    parser.add_argument("-t", "--target", default="Ar")
    parser.add_argument("--train-resizing", type=str, default="default")
    parser.add_argument("--val-resizing", type=str, default="default")
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=2048, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=45, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-e', '--eval-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    # parser.add_argument('--z_dim', type=int, default=128, metavar='N')
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--z_s1_dim', type=int, default=2, metavar='N')
    parser.add_argument('--z_s2_dim', type=int, default=32, metavar="N")
    parser.add_argument('--z_c1_dim', type=int, default=128, metavar="N")
    parser.add_argument('--z_c2_dim', type=int, default=2, metavar='N')

    parser.add_argument('--hidden_dim', type=int, default=4096, metavar='N')
    parser.add_argument("--u_dim", type=int, default=12, metavar="N")
    parser.add_argument('--beta', type=float, default=1., metavar='N')
    parser.add_argument('--name', type=str, default='', metavar='N')
    parser.add_argument('--init_value', type=float, default=0.0, metavar='N')
    parser.add_argument('--net', type=str, default='dirt', metavar='N')
    parser.add_argument('--lambda_vae', type=float, default=1e-3, metavar='N')
    parser.add_argument('--lambda_cls', type=float, default=1., metavar='N')
    parser.add_argument("--lambda_dom", type=float, default=0.1, metavar='N')
    parser.add_argument('--lambda_ent', type=float, default=0.1, metavar='N')
    parser.add_argument('--lambda_sem', type=float, default=0.1, metavar="N")
    parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N')
    parser.add_argument('--C_max', type=float, default=20., metavar='N')
    parser.add_argument('--C_stop_iter', type=int, default=10000, metavar='N')
    parser.add_argument("--is_use_tgt_shift", type=bool, default=True)
    parser.add_argument("--is_label_conditional_alignment",
                        type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--decay", type=float, default=0.3)
    parser.add_argument("--class_num", type=int, default=65)
    parser.add_argument("--mcc_weight", type=float, default=1.0)

    args = parser.parse_args()
    base_root = "/l/users/da_exp_ent"
    z_info = f"{args.z_s1_dim}-{args.z_s2_dim}-{args.z_c1_dim}-{args.z_c2_dim}"
    args.name = f"tgt-{args.target}-seed-{args.seed}-lr-{args.lr}-lambda_dom-{args.lambda_dom}-entropy_thr-{args.entropy_thr}"
    model_id = f"{args.data}_{args.target}/{args.name}-lam_vae_{args.lambda_vae}-lambda_ent_{args.lambda_ent}-{z_info}-is_tgt-{args.is_use_tgt_shift}-{args.is_label_conditional_alignment}-{args.lambda_sem}"
    args.log = os.path.join(base_root, args.log, model_id)

    args.z_dim = args.z_s1_dim + args.z_s2_dim + args.z_c1_dim + args.z_c2_dim

    args.source = [i for i in args.source.split(',')]
    args.target = [i for i in args.target.split(',')]
    args.n_domains = len(args.source) + len(args.target)
    args.input_dim = 2048

    if 'pacs' in args.root:
        args.input_dim = 512
        args.hidden_dim = 256

    args.norm_id = args.n_domains - 1
    wandb.init(
        project="domain_adaptation_subspace_identifiability",
        group=args.name,
    )
    wandb.config.update(args)

    main(args)
    flag = open(os.path.join(args.log, "finish_flag.txt"), "w")
    flag.close()
