dataset = "voc"
evo = False
if dataset == "coco":
    data_root = "/home/wangxucong/datasets/coco"
    batch_size = 64
    max_epoch = 80
    lr = 1e-1
elif dataset == 'voc':
    data_root = "/home/wangxucong/datasets/VOC2007"
    batch_size = 64
    max_epoch = 80 
    lr =  1e-2
elif dataset == 'fli':
    data_root = "/home/wangxucong/datasets/frieren"
    batch_size = 128
    max_epoch = 80 
    lr = 1e-2
lrp = 0.1
opacity = 0.6
pretrained = True
img_size = 224 
patch = 7
heads = 2
attn_dropout = 0.4    
iteration = 3
step = 2