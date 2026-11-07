# -*- coding: utf-8 -*-
"""Inference with exactly the context resizing and normalization used in training."""
import cv2
import numpy as np
from finetune_negatives import norm_patch


def context_probability(detector,image,sizes=(32,),copy_red=False,batch=128):
    image=np.asarray(image,np.float32).copy()
    if copy_red: image[:,:,3]=image[:,:,2]
    h,w=image.shape[:2]; total=np.zeros((h,w),np.float32)
    for size in sizes:
        step=size//2; halo=size//2
        padded=np.pad(image,((halo,halo+size),(halo,halo+size),(0,0)),mode='reflect')
        value=np.zeros((h,w),np.float32); counts=np.zeros((h,w),np.float32)
        coords=[(y,x) for y in range(0,h,step) for x in range(0,w,step)]
        for start in range(0,len(coords),batch):
            subset=coords[start:start+batch]; features=[]
            for cy,cx in subset:
                # Context is centred at original (cx,cy), including reflected
                # margin when needed. Counted outputs are clipped to real pixels.
                patch=padded[cy:cy+size,cx:cx+size]
                if size!=32: patch=cv2.resize(patch,(32,32),interpolation=cv2.INTER_LINEAR if size<32 else cv2.INTER_AREA)
                features.append(patch.transpose(2,0,1))
            prob=detector.predict_batch(norm_patch(np.stack(features)))
            for (cy,cx),p in zip(subset,prob):
                if size!=32: p=cv2.resize(p,(size,size),interpolation=cv2.INTER_AREA if size<32 else cv2.INTER_LINEAR)
                y0,x0=cy-halo,cx-halo; y1,x1=y0+size,x0+size
                a,b=max(0,y0),max(0,x0); c,d=min(h,y1),min(w,x1)
                value[a:c,b:d]+=p[a-y0:c-y0,b-x0:d-x0]; counts[a:c,b:d]+=1
        if (counts==0).any(): raise ValueError('Inference did not cover every observed grid pixel')
        total+=value/counts
    return total/len(sizes)
