export CUDA_VISIBLE_DEVICES=0,3,6,7 # #2,5,6,7 # 0,1,4,5
export WORLD_SIZE=4
export OMP_NUM_THREADS=1

# train 
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25644  kd.py --udec nnn --ngpu ${WORLD_SIZE}   --world_size  ${WORLD_SIZE}

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25643  kd.py --udec nnn --ngpu ${WORLD_SIZE}   --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25649  kd.py --udec sd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25639  kd.py --udec clv --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25601  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25501  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25500  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25500  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 
 
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25225  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 
  
 
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25128  kd.py --udec ddgsd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25178  kd.py --udec ddgsd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25118  kd.py --udec ddgsd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25018  kd.py --udec ddgsd --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} 


# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25266  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} 
 



# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24718  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 22018  kd.py --udec pskd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 
  
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24028  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24039  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24058  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 


# pretrain for ud

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24153  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24154  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24155  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} --para 1


# gnn layer 
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24156  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} --para 1
  
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24158  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24158  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} --para 1

python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port ${RANDOM%100+24000}  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24159  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24159  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24160  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24161  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24057  kd.py --udec sd --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} --para 1

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24059  kd.py --udec mulsupcon --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24059  kd.py --udec mulsupcon --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24068  kd.py --udec sd --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24068  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 23298  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 


# python  kd.py --udec sd --model resnet34

# python  kd.py --udec uskd --model resnet34    

# python  kd.py --udec sd --model mobilenet_v2    

# python  kd.py --udec uskd --model mobilenet_v2
    
# python  kd.py --udec dlb --model mobilenet_v2   



# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24358  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 22058  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 21058  kd.py --udec sd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 




# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24018  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24018  kd.py --udec pskd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 
  
# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24028  kd.py --udec dlb --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24039  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 24058  kd.py --udec clv --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 



# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25229  kd.py --udec uskd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25498  kd.py --udec sd --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25598  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25510  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model mobilenet_v2  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25609  kd.py --udec nnn --ngpu ${WORLD_SIZE} --model swin_t  --world_size  ${WORLD_SIZE} 

# python -m torch.distributed.launch  --nproc_per_node ${WORLD_SIZE}  --master_port 25649  kd.py --udec sd --ngpu ${WORLD_SIZE} --model resnet34  --world_size  ${WORLD_SIZE} 

# visualize
# python -m torch.distributed.launch  --nproc_per_node 4  --master_port 25642  kd.py --udec sd --ngpu 4  --world_size  ${WORLD_SIZE} 