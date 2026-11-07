"""Explicit reference-scale input route;not evidence of new sensor accuracy."""
import math
import numpy as np
import cv2


class SpatialInputContract:
    def __init__(self,image,valid,meters_per_pixel,mode='native'):
        self.source_shape=image.shape[:2];self.mode=mode
        self.source_mpp=meters_per_pixel
        self.image=image;self.valid=np.asarray(valid,bool)
        self.native_support=self.valid.copy();self.min_region_area=10
        self.record={'mode':mode,'source_shape':list(self.source_shape),'analysis_shape':list(self.source_shape),
            'source_meters_per_pixel':meters_per_pixel,'resampled':False,
            'accuracy_on_real_high_resolution_sensors_verified':False,
            'ground_boundary_precision_verified':False}
        if mode=='native':return
        if mode!='reference_10m':raise ValueError('analysis_grid只支持native或reference_10m')
        if meters_per_pixel is None or not math.isfinite(meters_per_pixel) or meters_per_pixel<=0:
            raise ValueError('参考尺度分析需要文件中的米/像素或明确提供的分辨率，不能猜测地面尺度')
        if meters_per_pixel>10.01:
            raise ValueError('影像分辨率粗于10米，不能通过插值补足参考尺度所需的细节')
        h,w=self.source_shape;ah,aw=int(round(h*meters_per_pixel/10)),int(round(w*meters_per_pixel/10))
        if min(ah,aw)<32:
            raise ValueError('参考尺度模式需要至少约320×320米的原始范围，请扩大影像上下文')
        self.record.update(analysis_shape=[ah,aw],reference_meters_per_pixel=10,
            effective_pixel_size_m=[w*meters_per_pixel/aw,h*meters_per_pixel/ah],
            source_observed_pixels=int(self.valid.sum()),
            basis='consistency with already verified10mdevelopment inputs;not a new sensor calibration',
            display_grid='original input grid;coarse predictions projected back without adding boundary detail')
        if (ah,aw)!=(h,w):
            coverage=cv2.resize(self.valid.astype(np.float32),(aw,ah),interpolation=cv2.INTER_AREA)
            averaged=cv2.resize(image*self.valid[:,:,None],(aw,ah),interpolation=cv2.INTER_AREA)
            self.valid=coverage>=.999999
            self.image=np.ascontiguousarray(averaged/np.maximum(coverage[:,:,None],1e-6),np.float32)
            self.image[~self.valid]=0
            self.native_support=(cv2.resize(self.valid.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0)&valid
            self.min_region_area=int(math.ceil(10*(h/ah)*(w/aw)))
            self.record['resampled']=True
        self.record.update(inference_supported_native_pixels=int(self.native_support.sum()),
            inference_excluded_native_pixels=int(np.asarray(valid,bool).sum()-self.native_support.sum()),
            min_reported_region_native_pixels=self.min_region_area,
            note='按约10米参考格网分析；显示回原图不提高边界精度，真实高分辨率识别效果仍待独立验收')
        if not self.valid.any():raise ValueError('参考格网没有完整可用的观测单元，请扩大有效区域或换用清晰影像')

    def restore_probability(self,probability):
        if probability.shape!=self.image.shape[:2]:raise ValueError('模型概率范围与参考格网不一致')
        if probability.shape==self.source_shape:return probability
        return cv2.resize(probability,(self.source_shape[1],self.source_shape[0]),interpolation=cv2.INTER_NEAREST)
