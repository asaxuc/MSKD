import torch


class PipeLine():
    def __init__(self,):
        pass

    def get_template(self, x, bbox):
        temp = torch.ones_like(x).cuda()
        i, j, h ,w = bbox 
        temp[:,:,i:i+h,j:j+w] = 0
        return (~(temp == 0)).to(torch.int),  ((temp == 0)).to(torch.int)
 
    def pipe_1(self, x, iteration, bbox, opacity):
        # for rare. x: input image + ROI, size:  (iteration+1)*B C H W
        # ret: opacity-changed image + ROI, size:  (iteration+1)*B C H W
         
        realx = x   
        for bbx in bbox:  # x[B*a:B*a+B]
            mm, fmm = self.get_template(realx, bbx)
            realx = realx * (mm*opacity + fmm)  

        return realx
 
    def pipe_2(self, x, iteration, bbox):
        # all for else. x: input image
        # ret: image  
        aug = []
        for bbx in bbox: # x[B*a:B*a+B]
            mm, fmm = self.get_template(x, bbx)
            aug.append(x * (fmm))  

        return torch.cat(aug,dim=0) 
    
    def forward(self, k,  stage=1, **kwargs):
        if stage == 2:
            if 'idtr' in kwargs.keys() and kwargs['idtr']:
                k = self.pipe_1(k, kwargs["iteration"], kwargs["bbox"],kwargs["opacity"]).detach()
            else:
                k = self.pipe_2(k, kwargs["iteration"], kwargs["bbox"]).detach()
        
        return k