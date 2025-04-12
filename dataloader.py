import os
from torch.utils.data import DataLoader
import torch.nn as nn
from data import VocDataset, CocoDataset, NUSWideDataset, FlickrDataset
import torch


def voc(cfg, data_root, args):

    train_img_prefix = os.path.join(data_root, "VOCtrainval2007/VOCdevkit/VOC2007")
    train_ann_file = os.path.join(data_root, "VOCtrainval2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt")
    test_img_prefix = os.path.join(data_root, "VOCtest2007/VOCdevkit/VOC2007")
    test_ann_file = os.path.join(data_root, "VOCtest2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt")

    class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                  "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    if not args.use_parallel:
        train_dataset = VocDataset(args, train_img_prefix, train_ann_file, class_name, img_size=cfg.img_size, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=8, drop_last=True, pin_memory=True)
        train_loader.num_classes = len(train_dataset.CLASSES)
        print(">>> Train Dataloader Built!")

        test_dataset = VocDataset(args, test_img_prefix, test_ann_file, class_name, img_size=cfg.img_size, train_mode=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)
        test_loader.num_classes = len(test_dataset.CLASSES)
        print(">>> Test Dataloader Built!")
    else:
        train_dataset = VocDataset(args, train_img_prefix, train_ann_file, class_name, img_size=cfg.img_size, train_mode=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_dataset = VocDataset(args, test_img_prefix, test_ann_file, class_name, img_size=cfg.img_size, train_mode=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=int(cfg.batch_size / args.ngpu), sampler=train_sampler, num_workers=8, drop_last=True, pin_memory=True)
        train_loader.num_classes = len(train_dataset.CLASSES)
        test_loader = DataLoader(test_dataset, batch_size=int(cfg.batch_size / args.ngpu), sampler=test_sampler, num_workers=8)
        test_loader.num_classes = len(test_dataset.CLASSES)
        print(">>> Train and Test Dataloader Built!")
        

    return train_loader, test_loader


def flickr(cfg, data_root, args):
    train_img_prefix = data_root
    train_ann_file = None
    test_img_prefix = data_root
    test_ann_file = None

    class_name = ["sky", 'clouds', 'water', 'sea', 'river', 'lake', 'people', 'portrait','male', 'female',
                  'baby', 'night', 'plant_life', 'tree', 'flower', 'animals','dog', 'bird', 'structures', 'sunset', 'indoor', 'transport', 'car']

    if not args.use_parallel:
        train_dataset = FlickrDataset(args, train_img_prefix, train_ann_file, class_name, img_size=cfg.img_size,
                                      train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True, pin_memory=True)
        train_loader.num_classes = len(train_dataset.CLASSES)
        print(">>> Train Dataloader Built!")

        test_dataset = FlickrDataset(args, test_img_prefix, test_ann_file, class_name, img_size=cfg.img_size,
                                     train_mode=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)
        test_loader.num_classes = len(test_dataset.CLASSES)
        print(">>> Test Dataloader Built!")
    else:
        train_dataset = FlickrDataset(args, train_img_prefix, train_ann_file, class_name, img_size=cfg.img_size,
                                      train_mode=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_dataset = FlickrDataset(args, test_img_prefix, test_ann_file, class_name, img_size=cfg.img_size,
                                     train_mode=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=int(cfg.batch_size / args.ngpu), sampler=train_sampler,
                                  num_workers=8, drop_last=True, pin_memory=True)
        train_loader.num_classes = len(train_dataset.CLASSES)
        test_loader = DataLoader(test_dataset, batch_size=int(cfg.batch_size / args.ngpu), sampler=test_sampler,
                                 num_workers=8)
        test_loader.num_classes = len(test_dataset.CLASSES)
        print(">>> Train and Test Dataloader Built!")

    return train_loader, test_loader



def coco(cfg, data_root, args):

    train_img_prefix = os.path.join(data_root, "train2014")
    train_ann_file = os.path.join(data_root, "train_anno.json") 
    test_img_prefix = os.path.join(data_root, "val2014")
    test_ann_file = os.path.join(data_root, "val_anno.json")  

    class_name = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                  "train", "truck", "boat", "traffic_light", "fire_hydrant",
                  "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                  "skis", "snowboard", "sports_ball", "kite", "baseball_bat",
                  "baseball_glove", "skateboard", "surfboard", "tennis_racket",
                  "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl",
                  "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                  "hot_dog", "pizza", "donut", "cake", "chair", "couch",
                  "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
                  "mouse", "remote", "keyboard", "cell_phone", "microwave",
                  "oven", "toaster", "sink", "refrigerator", "book", "clock",
                  "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"]
    class_name.sort()

    if not args.use_parallel:
        train_dataset = CocoDataset(train_img_prefix, train_ann_file, class_name, img_size=cfg.img_size, train_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=8, drop_last=True)
        train_loader.num_classes = len(train_dataset.CLASSES)
        print(">>> Train Dataloader Built!")

        test_dataset = CocoDataset(test_img_prefix, test_ann_file, class_name, img_size=cfg.img_size, train_mode=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
        test_loader.num_classes = len(test_dataset.CLASSES)
        print(">>> Test Dataloader Built!")
    else:
        train_dataset = CocoDataset(train_img_prefix, train_ann_file, class_name, img_size=cfg.img_size, train_mode=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=8, drop_last=True)
        train_loader.num_classes = len(train_dataset.CLASSES)
        print(">>> Train Dataloader Built!")

        test_dataset = CocoDataset(test_img_prefix, test_ann_file, class_name, img_size=cfg.img_size, train_mode=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) 
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, sampler=test_sampler, num_workers=8)
        test_loader.num_classes = len(test_dataset.CLASSES)
        print(">>> Test Dataloader Built!")

    return train_loader, test_loader


def nus(cfg, data_root, args):
    img_prefix = f"{data_root}/Flickr"
    ann_prefix = data_root

    train_dataset = NUSWideDataset(img_prefix, ann_prefix, img_size=cfg.img_size,
                                   train_mode=True, ls=cfg.ls if hasattr(cfg, "ls") else 0.0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=8, drop_last=True)
    train_loader.num_classes = len(train_dataset.CLASSES)
    print(">>> Train Dataloader Built!")

    test_dataset = NUSWideDataset(img_prefix, ann_prefix, img_size=cfg.img_size, train_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    test_loader.num_classes = len(test_dataset.CLASSES)
    print(">>> Test Dataloader Built!")

    return train_loader, test_loader



DATA = {"coco":coco,
        "voc":voc,
        "nus":nus,
        "fli":flickr}