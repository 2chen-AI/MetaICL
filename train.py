# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os    #文件和目录操作
import argparse    #命令行参数解析
import pickle as pkl    #数据序列化
import random
import torch    #深度学习框架
import math
import json    #处理JSON数据
import string
import logging
import numpy as np

from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, AutoTokenizer    #处理文本标记化

from metaicl.data import MetaICLData    #自定义模块
from metaicl.model import MetaICLModel
from utils.data import load_data

def main(logger, args):    #日志记录器logger和命令行参数args
    #选择合适的分词器
    if args.gpt2.startswith("gpt2"):    #如果命令行参数args.gpt2以"gpt2"开头就是用GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    else:    #否则使用AutoTokenizer，并指定"gpt2"作为预训练模型
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    #设置训练参数
    batch_size = args.batch_size    #批次大小
    max_length_per_example = 256    #每个示例的长度和总最大长度
    max_length = 256
    if args.use_demonstrations:
        max_length = min(max_length * args.k, 1024)

    #加载训练数据
    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.batch_size, max_length, max_length_per_example))

    train_data = load_data(args.task, "train", args.k, seed=args.seed)
    #使用load_data函数加载指定任务(args.task)的训练数据，传入训练模式"train"、参数args.k和随机种子args.seed

    #统计训练数据中每个任务的数量
    train_counter = Counter()
    for dp in train_data:
        train_counter[dp["task"]] += 1
    if args.local_rank <= 0:    #可能和分布式训练有关
        for k, v in train_counter.items():    #将每个任务及其数量记录到日志中
            logger.info("[Train] %s\t%d" % (k, v))
        logger.info("%s on %s (%d train)" % (args.method, args.task, len(train_counter)))    #并记录训练方法、训练名称和训练任务总数

    if args.init_checkpoint is not None:
        assert os.path.exists(args.init_checkpoint)    #确保检查点文件存在

    ######### load tensorize data    #数据张量化
    metaicl_data = MetaICLData(logger, tokenizer, args.method, args.use_demonstrations,    #创建MetaICLData对象，传入相关参数
                               args.test_k, max_length, max_length_per_example,
                               do_tensorize=args.do_tensorize,
                               tensorize_dir=args.tensorize_dir,
                               n_process=args.n_process, n_gpu=args.n_gpu, local_rank=args.local_rank)
    metaicl_data.tensorize_for_training(train_data, keyword=args.task, seed=args.seed,    #调用tensorize_for_training对训练数据进行张量化处理，可选择使用随机英语单词args.use_random_english_words
                                        use_random_english_words=args.use_random_english_words)

    if args.do_tensorize:    #只进行数据张量化
        return

    #训练相关的随机种子设置和训练参数计算
    ######## actual training part

    #设置训练相关的随机种子，以确保可重复性
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.train_seed)

    num_training_steps = args.num_training_steps
    save_period = 5000
    log_period = 5000

    if args.no_masking:    #处理掩码
        metaicl_data.tensorized_inputs["token_type_ids"] = torch.ones_like(metaicl_data.tensorized_inputs["input_ids"])
    metaicl_data.print_tensorized_example()

    logger.info(args.out_dir)

    if args.local_rank<=0 and not os.path.exists(args.out_dir):    #记录输出目录信息
        os.makedirs(args.out_dir)

    metaicl_model = MetaICLModel(logger, args.out_dir, args.fp16, args.local_rank)    #创建MetaICLModel对象
    metaicl_model.load(args.init_checkpoint, args.gpt2)    #加载预训练模型
    metaicl_model.to_device()    #将模型移动到合适的设备(GPU或CPU)
    metaicl_model.setup_optimizer(args.optimization, num_training_steps, args.lr,    #设置优化器
                                  args.weight_decay, args.warmup_steps)
    metaicl_model.parallel()    #模型并行化设置
    metaicl_model.train()    #将模型设置为训练模式
    metaicl_model.do_train(metaicl_data, args.batch_size, num_training_steps, save_period, log_period)    #开始训练

if __name__=='__main__':

    parser = argparse.ArgumentParser()    #用于解析命令行参数
    parser.add_argument("--do_tensorize", default=False, action="store_true")
    parser.add_argument("--tensorize_dir", type=str, default="tensorized")
    parser.add_argument("--n_gpu", type=int, default=8)
    parser.add_argument("--n_process", type=int, default=40)

    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--test_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=30000)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_masking", default=False, action="store_true")
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--optimization", type=str, default="adamw")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
