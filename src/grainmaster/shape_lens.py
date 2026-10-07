"""Diagnostic plumb-line correction using independently detected outer edges.

No ticks, physical dimensions, rounded fits or equality of object sizes enter
the fit. Straightness alone cannot identify perspective or absolute scale.
The fixed image centre and zoom law are assumptions, not camera intrinsics.
"""
import cv2
import numpy as np
from scipy.optimize import minimize_scalar


def extract_shape_edges(outlines):
    """Select original contour samples on central straight portions only."""
    edges = []
    for name in ('ruler', 'card'):
        outline = outlines[name]
        points = np.asarray(outline['contour'], float).reshape(-1, 2)
        corners = np.asarray(outline['quadrilateral'], float).reshape(4, 2)
        vectors = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(vectors, axis=1)
        ts = np.column_stack([(points-a)@v/(v@v) for a,v in zip(corners,vectors)])
        distances = np.column_stack([
            np.linalg.norm(points-(a+np.clip(t,0,1)[:,None]*v),axis=1)
            for a,v,t in zip(corners,vectors,ts.T)])
        assigned = distances.argmin(axis=1)
        selected = np.argsort(lengths)[-2:] if name == 'ruler' else range(4)
        for i in selected:
            # Ruler ends/handle are outside the central body; card corners excluded.
            low, high = (.15,.85) if name == 'ruler' else (.15,.85)
            mask = (assigned == i)&(ts[:,i]>=low)&(ts[:,i]<=high)
            edge_points = points[mask]
            edge_points = edge_points[np.argsort(ts[mask,i])]
            if len(edge_points) >= 20:
                edges.append(dict(object=name, edge_index=int(i), points=edge_points))
    return edges


def transform_shape_points(points, model, zoom=1):
    """Observed -> candidate corrected pixels, direct inverse radial polynomial."""
    if not np.isfinite(zoom) or zoom <= 0:
        raise ValueError('zoom must be finite and positive')
    p = np.asarray(points, float)
    if p.shape[-1] != 2 or not np.isfinite(p).all():
        raise ValueError('points must be finite with last dimension 2')
    center = np.asarray(model['center_px'], float)
    scale = float(model['normalization_px'])
    u = (p-center)/scale
    radial = np.sum(u*u,axis=-1)
    factor = 1+float(model['k1'])/zoom**2*radial
    return center+scale*u*factor[...,None]


def _normal_residual(points):
    p = np.asarray(points, np.float32).reshape(-1,2)
    if len(p) < 3:
        raise ValueError('Each edge requires at least three points')
    vx,vy,x,y = cv2.fitLine(p,cv2.DIST_HUBER,0,.001,.001).ravel()
    return (p-np.array([x,y]))@np.array([-vy,vx])


def shape_edge_metrics(edges, model=None, zoom=1):
    values=[]
    for edge in edges:
        points=np.asarray(edge['points'],float)
        if model is not None:
            points=transform_shape_points(points,model,zoom)
        residual=_normal_residual(points)
        values.append(dict(object=edge['object'],edge_index=int(edge['edge_index']),
            point_count=len(points),rms_px=float(np.sqrt(np.mean(residual**2))),
            median_abs_px=float(np.median(np.abs(residual))),
            p95_abs_px=float(np.percentile(np.abs(residual),95))))
    return dict(edges=values,edge_count=len(values),
                mean_edge_rms_px=float(np.mean([v['rms_px'] for v in values]))
                if values else None)


def fit_shape_lens(records, image_size, bounds=(-.25,.25), max_points=400):
    """Fit base k1 on alternate original samples with equal weight per edge.

    image_size is (width,height). Digital zoom is modelled as a central crop and
    uniform rescale: inverse coefficient is base k1 / zoom**2. This is conditional
    on shared camera/focal setting and may fail for in-camera geometric processing.
    The other alternating samples are reported as within-edge holdout, which is
    weaker than held-out photographs because neighbouring pixels are correlated.
    """
    width,height=map(int,image_size)
    model=dict(k1=0.,center_px=[(width-1)/2,(height-1)/2],
               normalization_px=float(max(width,height)),image_size=[width,height],
               zoom_law='effective_k1 = base_k1 / digital_zoom_ratio**2',
               method='inverse_radial_plumb_line_k1_fixed_image_center',
               applied=False,physical_calibration=False)
    samples=[]
    held=[]
    for record in records:
        zoom=float(record.get('zoom',1))
        if not np.isfinite(zoom) or zoom<=0:
            raise ValueError('Invalid digital zoom')
        for edge in record['edges']:
            p=np.asarray(edge['points'],float)
            if len(p)<20 or not np.isfinite(p).all():
                continue
            train=p[::2]
            if len(train)>max_points:
                train=train[np.linspace(0,len(train)-1,max_points).astype(int)]
            samples.append((train,zoom))
            held.append((p[1::2],zoom))
    if not samples:
        raise ValueError('No sufficiently dense measured edges')
    def objective(k, data):
        candidate=dict(model,k1=float(k))
        costs=[]
        for p,z in data:
            r=_normal_residual(transform_shape_points(p,candidate,z))
            # Soft-L1 with 1 px transition protects boundary tracing outliers.
            costs.append(float(np.mean(2*(np.sqrt(1+r*r)-1))))
        return float(np.mean(costs))
    result=minimize_scalar(lambda k:objective(k,samples),bounds=bounds,
                           method='bounded',options={'xatol':1e-7})
    model.update(k1=float(result.x),status='diagnostic_only',edge_count=len(samples),
                 train_image_ids=[r['image_id'] for r in records],bounds=list(bounds),
                 bound_hit=bool(min(result.x-bounds[0],bounds[1]-result.x)<.005),
                 train_loss_before=objective(0,samples),train_loss_after=objective(result.x,samples),
                 alternating_holdout_loss_before=objective(0,held),
                 alternating_holdout_loss_after=objective(result.x,held),
                 limitations=['No physical dimensions or tick intervals used',
                 'Straightness cannot correct planar perspective or establish millimetres',
                 'Object bending and contour detection bias can imitate lens curvature',
                 'Fixed centre and central-crop digital zoom law are unverified assumptions'])
    model['loss_profile']=[dict(k1=float(k),loss=objective(k,samples))
                           for k in np.linspace(bounds[0],bounds[1],21)]
    return model


def remap_shape_lens(image, model, zoom=1):
    """Render candidate by numerically inverting direct correction for cv2.remap."""
    height,width=image.shape[:2]
    if not np.isfinite(zoom) or zoom<=0:
        raise ValueError('Invalid zoom')
    center=np.asarray(model['center_px'],float)
    scale=float(model['normalization_px'])
    k=float(model['k1'])/zoom**2
    # A conservative bound ensures radial derivative positive across frame/corners.
    corner_r2=max(np.sum(((np.array(p)-center)/scale)**2)
                  for p in [(0,0),(width-1,0),(0,height-1),(width-1,height-1)])
    if 1+3*k*corner_r2<=.2:
        raise ValueError('Radial map not safely invertible on frame')
    output=np.empty_like(image)
    for top in range(0,height,256):
        bottom=min(top+256,height)
        yy,xx=np.mgrid[top:bottom,0:width].astype(np.float64)
        tx=(xx-center[0])/scale;ty=(yy-center[1])/scale
        target=np.sqrt(tx*tx+ty*ty)
        radius=target.copy()
        for _ in range(12):
            radius-=(radius+k*radius**3-target)/(1+3*k*radius**2)
        ratio=np.divide(radius,target,out=np.ones_like(radius),where=target>0)
        mx=(center[0]+scale*tx*ratio).astype(np.float32)
        my=(center[1]+scale*ty*ratio).astype(np.float32)
        if not np.isfinite(mx).all() or not np.isfinite(my).all():
            raise ValueError('Non-finite inverse map')
        output[top:bottom]=cv2.remap(image,mx,my,cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
    return output
