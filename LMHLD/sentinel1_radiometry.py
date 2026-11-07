"""Original GRD DN to denoised sigma0; this performs no terrain/geometric correction."""
import numpy as np
import xml.etree.ElementTree as ET


def xml_root(path):
    root=ET.parse(path).getroot()
    for node in root.iter():node.tag=node.tag.split('}')[-1]
    return root


def numbers(node,name):
    text=node.findtext(name)
    if text is None:raise ValueError('Required original radar LUT missing:'+name)
    result=np.fromstring(text,sep=' ',dtype=np.float64)
    if not len(result) or not np.isfinite(result).all():raise ValueError('Invalid radar LUT:'+name)
    return result


def interpolated_range_vectors(nodes,value_name,rows,columns):
    nodes=sorted(nodes,key=lambda x:float(x.findtext('line')))
    if len(nodes)<2:raise ValueError('At least two radar LUT line vectors required')
    lines=np.asarray([float(x.findtext('line')) for x in nodes]);values=[]
    if np.any(np.diff(lines)<=0) or rows.min()<lines[0] or rows.max()>lines[-1]:raise ValueError('Radar window outside original line LUT support')
    for node in nodes:
        pixels=numbers(node,'pixel');lut=numbers(node,value_name)
        if len(pixels)!=len(lut) or np.any(np.diff(pixels)<=0):raise ValueError('Malformed original range LUT')
        if columns.min()<pixels[0] or columns.max()>pixels[-1]:raise ValueError('Radar window outside original pixel LUT support')
        values.append(np.interp(columns,pixels,lut))
    values=np.stack(values);right=np.searchsorted(lines,rows,side='right');right=np.clip(right,1,len(lines)-1);left=right-1
    fraction=(rows-lines[left])/(lines[right]-lines[left])
    return values[left]*(1-fraction[:,None])+values[right]*fraction[:,None]


def azimuth_noise_factor(root,rows,columns):
    nodes=root.findall('.//noiseAzimuthVector')
    if not nodes:return np.ones((len(rows),len(columns)),np.float64)
    result=np.zeros((len(rows),len(columns)),np.float64);count=np.zeros(result.shape,np.uint8)
    for node in nodes:
        start_y,end_y=int(node.findtext('firstAzimuthLine')),int(node.findtext('lastAzimuthLine'))
        start_x,end_x=int(node.findtext('firstRangeSample')),int(node.findtext('lastRangeSample'))
        yy=np.flatnonzero((rows>=start_y)&(rows<=end_y));xx=np.flatnonzero((columns>=start_x)&(columns<=end_x))
        if not len(yy) or not len(xx):continue
        lines=numbers(node,'line');lut=numbers(node,'noiseAzimuthLut')
        if len(lines)!=len(lut) or np.any(np.diff(lines)<=0):raise ValueError('Malformed azimuth noise LUT')
        if rows[yy].min()<lines[0] or rows[yy].max()>lines[-1]:raise ValueError('Outside azimuth noise LUT support')
        coefficient=np.interp(rows[yy],lines,lut)
        result[np.ix_(yy,xx)]=coefficient[:,None];count[np.ix_(yy,xx)]+=1
    if not np.all(count==1):raise ValueError('Incomplete or overlapping original azimuth-noise coverage;no guessed fill')
    if not np.isfinite(result).all() or np.any(result<=0):raise ValueError('Invalid noise factors')
    return result


def calibrate_grd_sigma0(raw,original_window,calibration_xml,noise_xml,product_xml):
    raw=np.asarray(raw)
    if raw.ndim!=2 or not np.isfinite(raw).all():raise ValueError('Finite original two-dimensional GRD DN required')
    left,top,width,height=[int(x) for x in original_window]
    if raw.shape!=(height,width) or left<0 or top<0:raise ValueError('Original raster window coordinate mismatch')
    product=xml_root(product_xml);lines=int(product.findtext('.//numberOfLines'));samples=int(product.findtext('.//numberOfSamples'))
    if top+height>lines or left+width>samples:raise ValueError('COG orientation/dimensions do not match original XML;explicit index mapping required')
    flags={x.text.lower() for x in product.findall('.//thermalNoiseCorrectionPerformed')}
    if len(flags)!=1 or not flags<={'true','false'}:raise ValueError('Original thermal noise state must be explicit and consistent')
    already_corrected=next(iter(flags))=='true'
    rows=np.arange(top,top+height,dtype=np.float64);columns=np.arange(left,left+width,dtype=np.float64)
    calibration=xml_root(calibration_xml);sigma=interpolated_range_vectors(calibration.findall('.//calibrationVector'),'sigmaNought',rows,columns)
    if np.any(sigma<=0):raise ValueError('Positive calibration coefficients required')
    power=raw.astype(np.float64)**2
    removed=np.zeros(raw.shape,np.float64)
    if not already_corrected:
        noise=xml_root(noise_xml);modern=noise.findall('.//noiseRangeVector')
        if modern:
            removed=interpolated_range_vectors(modern,'noiseRangeLut',rows,columns)*azimuth_noise_factor(noise,rows,columns)
        else:
            old=noise.findall('.//noiseVector')
            removed=interpolated_range_vectors(old,'noiseLut',rows,columns)
        if not np.isfinite(removed).all() or np.any(removed<0):raise ValueError('Nonfinite or negative noise power')
        power=power-removed
    valid=(raw>0)&(power>0)&np.isfinite(power)
    calibrated=np.zeros(raw.shape,np.float32);calibrated[valid]=(power[valid]/sigma[valid]**2).astype(np.float32)
    db=np.full(raw.shape,-9999.,np.float32);db[valid]=(10*np.log10(calibrated[valid])).astype(np.float32)
    return calibrated,db,valid,{
        'units':'sigma_nought_linear_power_and10log10dB','source_DN_squared_before_noise_removal':True,
        'thermal_noise':'source_already_corrected_not_subtracted_again' if already_corrected else 'range_and_available_azimuth_noise_removed_using_originalLUT',
        'valid_samples':int(valid.sum()),'unknown_or_below_noise_samples':int((~valid).sum()),
        'median_raw_noise_power':float(np.median(removed)),
        'geometric_or_radiometric_terrain_correction_performed':False,
        'not_phase_coherence_or_ground_displacement':True,
    }
