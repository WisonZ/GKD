from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from LMDB_new import LMDB_imgWise
from funcs import *
from margin.CosineMarginProduct import MarginCosineProduct
from backbone.resnetSEIR import *
from torch import distributed as dist
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from backbone.mobilenet import get_mbf
from distillation.gkd import GKD
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

import time
import numpy as np
# fix random seed

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

logger = Logger('./gkd.log')####hongwai

# num_epoch = 25#12
num_epoch = 20
lr = 0.1
warmup_epoch = 4
num_classes = 85742
min_lr = 2.5e-7
warmup_lr = 2.5e-7

batch_size = 128
start_epoch = 0
total_batch = 0
same_seeds(1024)

teacher = SEResNet_IR(50, num_classes = num_classes)#
student =get_mbf(fp16=False,num_features=512, blocks=(1, 4, 6, 2), scale=2,num_classes=num_classes)

teacher.fc =MarginCosineProduct(512, num_classes, m=0.4, scale=64)
student.fc =MarginCosineProduct(512, num_classes, m=0.4, scale=64)

teacher_model_path = '../train_recognition/models/IR50/train_resnetIR50_4gpu_cosface.tmp'
checkpoint = torch.load(teacher_model_path, map_location='cpu')
teacher.load_state_dict(checkpoint['state_dict'], strict=True)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, student.parameters()),lr=lr, momentum=0.9, weight_decay = 5e-4, nesterov=True)
from timm.scheduler.cosine_lr import CosineLRScheduler

teacher = teacher.to(device)
student = student.to(device)

teacher = torch.nn.parallel.DistributedDataParallel(teacher,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
student = torch.nn.parallel.DistributedDataParallel(student,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

db_paths = [
   '/media/ssd4/train.lmdb',
            ]
filelist_paths = [
    '/media/ssd4/train_filelist.txt',
                  ]
dataset = LMDB_imgWise(db_paths, filelist_paths,
                    transform=transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                     ])
                           )
num_tasks = dist.get_world_size()
global_rank = dist.get_rank()
sampler_train = DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
train_loader= DataLoader(
        dataset, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
    )
n_iter_per_epoch = len(train_loader)
num_steps = int(num_epoch * n_iter_per_epoch)
warmup_steps = int(warmup_epoch * n_iter_per_epoch)

scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            #t_mul=1.,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

criterion_s = GKD()
criterion_s.to(device)


batch_time = AverageMeter()
load_time = AverageMeter()
top1 = AverageMeter()
running_loss = AverageMeter()


for epoch in range(start_epoch,start_epoch+num_epoch):  #

    batch_time = AverageMeter()
    load_time = AverageMeter()
    top1 = AverageMeter()
    running_loss_ce_student = AverageMeter()
    running_loss_kd_student = AverageMeter()
    running_loss = AverageMeter()
    end_time = time.time()

    student.train()
    teacher.eval()

    for batch_idx, (images,labels) in enumerate(train_loader):

        load_time.update(time.time() - end_time)
        images_s = images.to(device) # Bx112x112x3
        labels = labels.to(device) # Bx1 [0,cls]
        # student forward

        output_dict = student(images,labels)
        outputs_student = output_dict

        with torch.no_grad():
            outputs_teacher = teacher(images, labels)

        logit_tmp, loss_dict = criterion_s(outputs_student,outputs_teacher,labels,epoch)
        loss_ce_student = loss_dict['loss_ce']
        loss_kd_student = loss_dict['loss_kd']

        loss_student = loss_ce_student+loss_kd_student
        prec1, = compute_accuracy(outputs_student, labels, topk=(1,))

        top1.update(prec1.item())

        # student backward
        optimizer.zero_grad()
        #optimizer_teacher.zero_grad()

        loss_student.backward()

        optimizer.step()

        scheduler.step_update((epoch-start_epoch) * n_iter_per_epoch + batch_idx)

        running_loss_ce_student.update(loss_ce_student.item())
        running_loss_kd_student.update(loss_kd_student.item())


        batch_time.update(time.time() - end_time)
        end_time = time.time()
        total_batch += 1
        lr_now = optimizer.param_groups[0]['lr']

        if batch_idx % 100 == 0:
            if local_rank == 0:
                logger("Epoch-Iter [%d/%d][%d/%d] Time_tot/load [%f][%f] lr [%g] ce_s [%f] kd_s [%f] Prec@1 [%f] "%( #l
                    epoch + 1, num_epoch + start_epoch, batch_idx, len(train_loader), batch_time.avg,
                    load_time.avg, lr_now, running_loss_ce_student.avg ,running_loss_kd_student.avg,top1.avg,))
#
    if (epoch + 1) % 1 == 0:
        if local_rank == 0:
            torch.save({'state_dict': student.module.state_dict(),  # model.module.state_dict()
                        }, './models/gkd/train_gkd_student_epoch_' + str(
                epoch + 1 + start_epoch) + '_' + str(local_rank) + '.tmp')

        print('save model')
