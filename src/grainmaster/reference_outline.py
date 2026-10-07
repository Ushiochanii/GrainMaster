"""Image-supported silhouettes of the reference objects, independent of calibration boxes."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _shape(contour, method, **metadata):
    contour = np.asarray(contour, np.float32).reshape(-1, 2)
    # Fit each straight central side independently; do not impose parallelism.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    order = np.argsort(np.arctan2(box[:,1]-box[:,1].mean(), box[:,0]-box[:,0].mean()))
    box = box[order]
    lines=[]
    for p,q in zip(box,np.roll(box,-1,axis=0)):
        v=q-p; t=((contour-p)@v)/(v@v)
        d=np.abs((v[0]*(contour[:,1]-p[1])-v[1]*(contour[:,0]-p[0])))/np.linalg.norm(v)
        pts=contour[(t>.15)&(t<.85)&(d<max(12,min(rect[1])*.05))]
        if len(pts)<10: raise ValueError('Insufficient central boundary support')
        vx,vy,x,y=cv2.fitLine(pts,cv2.DIST_HUBER,0,.01,.01).ravel()
        lines.append((np.array([x,y]),np.array([vx,vy])))
    quad=[]
    for i in range(4):
        p,v=lines[i-1];q,w=lines[i]
        z=np.linalg.solve(np.column_stack([v,-w]),q-p)
        quad.append(p+z[0]*v)
    return {'contour':contour,'bbox':tuple(map(int,cv2.boundingRect(contour))),
            'quadrilateral':np.asarray(quad,np.float32),
            'metadata':{'method':method,'holes_ignored':True,'quad_definition':'intersection of independently fitted central side lines',**metadata}}


def _ruler(image,spatial):
    x,y,w,h=spatial.bbox
    pad=int(min(w,h)*.5)
    x0=max(0,x-pad);y0=max(0,y-pad)
    crop=image[y0:min(image.shape[0],y+h+pad),x0:min(image.shape[1],x+w+pad)]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    m=(cv2.GaussianBlur(gray,(5,5),0)>160).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates=[c for c in contours if cv2.contourArea(c)>w*h*.20]
    if not candidates: raise ValueError('No complete white ruler silhouette')
    c=max(candidates,key=cv2.contourArea).reshape(-1,2)+[x0,y0]
    return _shape(c,'bright exterior silhouette, internal printing/hole excluded from exterior')


def _card(image,color):
    grid=np.asarray(color.metadata['patch_centers'],np.float32).reshape(4,6,2)
    if grid[0,-1,0] < grid[0,0,0]: grid=grid[:,::-1]
    if grid[-1,0,1] < grid[0,0,1]: grid=grid[::-1]
    lattice=np.array([[x,y] for y in range(4) for x in range(6)],np.float32)
    H,_=cv2.findHomography(lattice,grid.reshape(-1,2))
    unit=float(np.median(np.linalg.norm(np.diff(grid,axis=1),axis=2)))
    u=min(180.,unit); origin=np.array([1.5,1.2]); size=(int(8.5*u),int(5.5*u))
    T=np.array([[u,0,origin[0]*u],[0,u,origin[1]*u],[0,0,1.]])@np.linalg.inv(H)
    warped=cv2.warpPerspective(image,T,size)
    gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY).astype(float)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    # Search actual dark printed outer boundary, in generous grid-relative bands.
    bands=[(-1.15,-.50),(6.12,6.4),(-.95,-.45),(3.35,3.95)]
    positions=[]; seams=[]
    for i,(a,b) in enumerate(bands):
        vertical=i<2; axis=0 if vertical else 1
        offset=origin[axis];lo=int((a+offset)*u);hi=int((b+offset)*u)
        if vertical:
            subset=gray[int((origin[1]+.2)*u):int((origin[1]+2.8)*u),lo:hi]
            profile=np.median(subset,axis=0)
        else:
            subset=gray[lo:hi,int((origin[0]+.2)*u):int((origin[0]+4.8)*u)]
            profile=np.median(subset,axis=1)
        loc=lo+int(np.argmin(gaussian_filter1d(profile,1)))
        positions.append(loc)
        # Narrow local minimum search follows perspective/curvature after grid alignment.
        search=int(u*.08);low=max(lo,loc-search);high=min(hi,loc+search+1)
        strip=gray[:,low:high] if vertical else gray[low:high,:].T
        # Prior prevents distant background marks attracting the boundary.
        cost=strip+0.07*(np.arange(low,high)-loc)**2
        seam=low+np.argmin(cost,axis=1)
        seams.append(gaussian_filter1d(seam.astype(float),1.5))
    yy,xx=np.indices(gray.shape)
    mask=((xx>=seams[0][:,None])&(xx<=seams[1][:,None])&
          (yy>=seams[2][None,:])&(yy<=seams[3][None,:])).astype(np.uint8)*255
    # Trace rounded corner arcs from actual border intensity along radial rays.
    radius=u*.18
    for xi,sx in [(positions[0],-1),(positions[1],1)]:
        for yi,sy in [(positions[2],-1),(positions[3],1)]:
            center=np.array([xi-sx*radius,yi-sy*radius])
            angles=np.linspace(0,np.pi/2,91)
            radii=np.linspace(radius*.50,radius*1.65,100)
            rayx=center[0]+sx*np.cos(angles)[:,None]*radii
            rayy=center[1]+sy*np.sin(angles)[:,None]*radii
            vals=cv2.remap(gray.astype(np.float32),rayx.astype(np.float32),rayy.astype(np.float32),cv2.INTER_LINEAR)
            cost=vals+0.025*(radii-radius)**2
            rr=gaussian_filter1d(radii[np.argmin(cost,axis=1)],2)
            dx=(xx-center[0])*sx;dy=(yy-center[1])*sy
            region=(dx>=0)&(dy>=0)&(dx<radius*1.7)&(dy<radius*1.7)
            theta=np.arctan2(np.maximum(0,dy),np.maximum(0,dx))
            limit=np.interp(theta.ravel(),angles,rr).reshape(theta.shape)
            mask[region]=(np.hypot(dx,dy)[region]<=limit[region]).astype(np.uint8)*255
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    c=max(cs,key=cv2.contourArea)
    raw=cv2.perspectiveTransform(c.astype(np.float32),np.linalg.inv(T)).reshape(-1,2)
    return _shape(raw,'grid-guided outer dark-border intensity tracing',
                  search_resolution_px=unit/u,rectified_side_positions=positions,
                  corner_note='dark-line radial corner tracing with 0.18 grid-spacing radius prior; inspect visually',corner_tracing=True)


def detect_reference_outlines(image_bgr, spatial, color):
    return {'ruler':_ruler(image_bgr,spatial),'card':_card(image_bgr,color)}


def refit_reference_outline(contour, method='transformed observed exterior contour'):
    """Refit independent central side lines after a point transformation.

    Retains the supplied dense silhouette; does not replace it with a rectangle.
    """
    return _shape(contour, method)


def draw_reference_outline_overlay(image_bgr,result):
    out=image_bgr.copy()
    for name,shape in result.items():
        c=shape['contour'].astype(np.int32).reshape(-1,1,2)
        cv2.polylines(out,[c],True,(0,255,255) if name=='card' else (0,80,255),3)
        x,y,w,h=shape['bbox'];cv2.putText(out,name,(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,255),3)
    return out





