(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const prefs=()=>window.GrainMasterSettings;
  const pref=(group,key,fallback)=>prefs()?.get(group,key,fallback) ?? fallback;
  const labelsEnabled=()=>prefs()?.paperLabelsEnabled() ?? false;
  function rememberPhoto(id){if(pref('general','remember_last_photo',true)&&id)try{localStorage.setItem('grainmaster:last-photo',id);}catch{}}
  const state = {result:null, labelJob:'idle', labelProgress:{}, selectedId:null, showPhoto:true, images:[], dishId:null, seedKey:null, mode:'overview', summaryMode:'draft', sidebarView:'review', image:null, corrected:null, correctedPromise:null, spatial:null, scale:1, x:0, y:0, drag:null, busy:false, loadToken:0, displayToken:0};
  let analysis=null, animationFrame=null;
  const canvas = $('photo-canvas'), ctx = canvas.getContext('2d');
  const fmt = (v, digits=2) => Number.isFinite(v) ? v.toFixed(digits) : '—';
  const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const selectedDish = () => state.result?.dishes.find(d => d.dish_id === state.dishId);
  const allSeeds = () => state.result?.dishes.flatMap(d => d.seeds) || [];
  const dishName = d => d?.sample_id || d?.label || '';
  let labelWatchToken=0;
  const selectedSeed = () => allSeeds().find(s => s.key === state.seedKey);
  const isPriority = s => s.review_state === 'pending' && (s.ambiguity_reasons || []).length > 0;
  const confidenceName = s => s.review_state==='discarded'?'Excluded':s.review_state==='confirmed'?'Checked':isPriority(s)?'Auto · Review':'Auto';
  const confidenceClass = s => s.review_state==='discarded'?'discarded':isPriority(s)?'low-confidence':'high-confidence';
  const seedColor = s => s.review_state==='discarded'?'#b2bdc4':isPriority(s)?'#ed5555':'#b68a50';
  let noticeTimer;
  function notice(message, error=false) { clearTimeout(noticeTimer); $('notice').textContent=message; $('notice').classList.toggle('error',error); $('notice').hidden=!message;if(message && !error && message.endsWith('Saved.'))noticeTimer=setTimeout(()=>notice(''),2200); }
  async function api(url, options={}) { const response=await fetch(url,options); if(!response.ok) { let detail; try { detail=(await response.json()).detail; } catch { detail=response.statusText; } throw new Error(typeof detail==='string'?detail:JSON.stringify(detail)); } return response.json(); }
  function busy(value) { state.busy=value; $('process-button').disabled=value || !state.selectedId; $('export-button').disabled=value||!state.result; $('image-select').disabled=value;document.querySelectorAll('[data-sample-input]').forEach(b=>b.disabled=value);$('read-paper-labels').disabled=value||!state.result||!labelsEnabled();$('tab-summary').disabled=value||!state.result;$('photo-search').disabled=value; for(const id of ['upload','upload-button','folder-upload','folder-button'])$(id).disabled=value; document.querySelectorAll('[data-image], [data-delete-image]').forEach(button=>button.disabled=value); }
  const imageWidth = img => Number(img.dataset.fullWidth) || img.naturalWidth;
  const imageHeight = img => Number(img.dataset.fullHeight) || img.naturalHeight;
  const displayImages=new Map();let fullImageQueue=Promise.resolve();
  function nativeImage(url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('The image could not be loaded. Check the processing result.'));img.src=url;});}
  function imageLoad(url) {
    const progressive=/^\/api\/images\/[^/]+\/(original|asset\/[^/?]+)(\?|$)/.test(url);
    if(!progressive)return nativeImage(url);
    if(displayImages.has(url))return displayImages.get(url).then(img=>{queueClearImage(url,img,state.loadToken);return img;});
    const token=state.loadToken;
    const loading=(async()=>{
      const previewUrl=new URL(url,location.origin);previewUrl.searchParams.set('preview','true');
      const response=await fetch(previewUrl);if(!response.ok)throw new Error('The image preview could not be loaded.');
      const blobUrl=URL.createObjectURL(await response.blob());let img;
      try{img=await nativeImage(blobUrl);}finally{URL.revokeObjectURL(blobUrl);}
      img.dataset.fullWidth=response.headers.get('X-Image-Width') || img.naturalWidth;
      img.dataset.fullHeight=response.headers.get('X-Image-Height') || img.naturalHeight;
      queueClearImage(url,img,token);
      return img;
    })();
    displayImages.set(url,loading);loading.catch(()=>displayImages.delete(url));
    while(displayImages.size>4)displayImages.delete(displayImages.keys().next().value);
    return loading;
  }
  function queueClearImage(url,img,token){
    if(img.dataset.clearReady || img.dataset.clearLoading)return;
    img.dataset.clearLoading='true';
    fullImageQueue=fullImageQueue.catch(()=>{}).then(async()=>{
      await new Promise(resolve=>setTimeout(resolve,500));
      try{
        if(token!==state.loadToken)return;
        const full=await nativeImage(url);
        img.onload=()=>{img.dataset.clearReady='true';if(token===state.loadToken)draw();};
        img.src=full.src;
      }catch{/* The lightweight preview remains usable. */}
      finally{delete img.dataset.clearLoading;}
    });
  }
  async function ensureCorrected(announce=false) {
    if(state.corrected)return state.corrected;
    if(state.correctedPromise)return state.correctedPromise;
    const token=state.loadToken;
    const control=$('corrected-control'), toggle=$('corrected-toggle'), spatial=$('spatial-toggle');
    control?.classList.add('loading');
    toggle.disabled=true;
    spatial.disabled=true;
    if(announce)notice('Generating global color-corrected preview…');
    state.correctedPromise=imageLoad(state.result.corrected_url)
      .then(img=>{
        if(token!==state.loadToken)return null;
        state.corrected=img;
        return img;
      })
      .finally(()=>{
        if(token===state.loadToken){
          state.correctedPromise=null;
          control?.classList.remove('loading');
          toggle.disabled=false;
          spatial.disabled=!state.result?.spatial_preview;
          if(announce)notice('');
        }
      });
    return state.correctedPromise;
  }
  function renderPhotos(){ $('photo-count').textContent=state.images.length; $('photo-list').innerHTML=state.images.filter(i=>i.filename.toLowerCase().includes($('photo-search').value.toLowerCase())).map(i=>`<div class="photo-item"><button class="photo-tile ${i.image_id===state.selectedId?'active':''} ${i.is_example?'example-photo':''}" data-image="${escape(i.image_id)}" aria-current="${i.image_id===state.selectedId?'true':'false'}" ${state.busy?'disabled':''}><img src="/api/images/${encodeURIComponent(i.image_id)}/thumbnail" alt="${escape(i.filename)}" loading="lazy"><span class="photo-caption"><strong>${escape(i.filename)}</strong><small>${i.is_example?'Example':i.processed?'Ready':'Preview'}</small></span></button>${i.is_imported?`<button type="button" class="photo-delete" data-delete-image="${escape(i.image_id)}" aria-label="Delete ${escape(i.filename)}" title="Delete photo and analysis results" ${state.busy?'disabled':''}>Delete</button>`:''}</div>`).join(''); }
  async function deleteImage(id){
    const item=state.images.find(i=>i.image_id===id);
    if(state.busy||!item?.is_imported)return;
    if(!window.confirm(`Delete “${item.filename}” and its analysis results? This cannot be undone.`))return;
    const index=state.images.indexOf(item), wasSelected=id===state.selectedId;
    busy(true);
    notice('Deleting photo…');
    try{
      await api(`/api/images/${encodeURIComponent(id)}`,{method:'DELETE'});
      for(const key of displayImages.keys())if(key.startsWith(`/api/images/${encodeURIComponent(id)}/`))displayImages.delete(key);
      try{if(localStorage.getItem('grainmaster:last-photo')===id)localStorage.removeItem('grainmaster:last-photo');}catch{}
      if(wasSelected){state.loadToken++;resetPhoto(null);}
      await refreshImages();
      if(wasSelected){
        const next=state.images[Math.min(index,state.images.length-1)];
        if(next)await selectImage(next.image_id);
      }
      notice('Photo deleted.');
    }catch(err){notice(`Could not delete photo: ${err.message}`,true);}
    finally{busy(false);}
  }
  async function refreshImages(preferred) { const response=await api('/api/images'); state.images=response.images; $('image-select').innerHTML=state.images.map(i=>`<option value="${escape(i.image_id)}">${escape(i.filename)}${i.processed?' · Analyzed':''}</option>`).join(''); const chosen=preferred || state.selectedId; if(chosen && state.images.some(i=>i.image_id===chosen))$('image-select').value=chosen; renderPhotos(); }
  function resetPhoto(id){endAnalysis();labelWatchToken++;state.labelJob='idle';state.labelProgress={};state.dishDrafts={};state.sidebarView='review';$('review-sidebar').hidden=false;$('measurement-sidebar').hidden=true;$('tab-inspect').classList.add('active');$('tab-summary').classList.remove('active');$('tab-inspect').setAttribute('aria-selected','true');$('tab-summary').setAttribute('aria-selected','false');$('paper-label-status').textContent='';$('paper-label-list').innerHTML=''; state.selectedId=id;state.result=null;state.image=null;state.corrected=null;state.correctedPromise=null;state.spatial=null;state.dishId=null;state.seedKey=null;state.mode='overview';state.showPhoto=true;state.displayToken++;$('corrected-toggle').checked=false;$('spatial-toggle').checked=false;for(const name of ['corrected-toggle','spatial-toggle'])$(name).disabled=true;$('mask-toggle').checked=pref('review','show_masks',true);$('ids-toggle').checked=pref('review','show_ids',true);$('seed-filter').value=pref('review','default_filter','all');$('image-select').value=id;renderPhotos();requestAnimationFrame(()=>document.querySelector("[data-image].active")?.scrollIntoView({block:"nearest"}));render(); }
  async function selectImage(id){ rememberPhoto(id); const item=state.images.find(i=>i.image_id===id);if(!item)return;if(item.processed)return loadResult(id); const token=++state.loadToken;resetPhoto(id);busy(true);notice('Opening photo preview…');try{const img=await imageLoad(`/api/images/${encodeURIComponent(id)}/original`);if(token!==state.loadToken)return;state.image=img;render();fit();notice('');}catch(err){if(token===state.loadToken)notice(`Preview failed: ${err.message}`,true);}finally{if(token===state.loadToken)busy(false);} }
  async function loadResult(id,{deferRender=false}={}) { const token=++state.loadToken;if(!deferRender){resetPhoto(id);busy(true);notice('Opening photo and measurement results…');} try { const result=await api(`/api/results/${encodeURIComponent(id)}`); const [original,corrected]=await Promise.all([imageLoad(result.image_url),imageLoad(result.corrected_url)]); if(token!==state.loadToken)return null; state.selectedId=id;rememberPhoto(id);state.result=result; state.image=original; state.corrected=corrected; state.correctedPromise=null; state.dishId=null;state.seedKey=null;state.mode='overview';state.showPhoto=true; $('corrected-toggle').checked=pref('general','default_image_layer','color')==='color'; $('seed-filter').value=pref('review','default_filter','all'); $('spatial-toggle').disabled=!result.spatial_preview; $('corrected-toggle').disabled=false; if(!deferRender){setSidebarView(pref('general','default_view','review'));render();fit();notice('');watchLabels(id);} return result; } catch(err) { if(token===state.loadToken&&!deferRender)notice(`Open failed: ${err.message}`,true); throw err; } finally { if(token===state.loadToken&&!deferRender)busy(false); } }
  async function importFiles(files){const allowed=[...files].filter(f=>/\.(jpe?g|png|tiff?|webp)$/i.test(f.name));if(!allowed.length){notice('Select JPG, PNG, TIFF, or WebP images.',true);return;}busy(true);const imported=[];const failures=[];try{for(let n=0;n<allowed.length;n++){notice(`Importing photo ${n+1}/${allowed.length}: ${allowed[n].name}`);const form=new FormData();form.append('file',allowed[n]);try{const info=await api('/api/upload',{method:'POST',body:form});imported.push(info.image_id);}catch(err){failures.push(`${allowed[n].name}: ${err.message}`);}}await refreshImages(imported[0]);if(imported.length)await selectImage(imported[0]);notice(`Imported ${imported.length} photos.${failures.length?` ${failures.length} import(s) failed: ${failures.join('; ')}`:''}`,failures.length>0);}finally{busy(false);$('upload').value='';$('folder-upload').value='';} }
  const analysisStages=[['ruler','Ruler'],['color','Color'],['dishes','Dishes'],['spatial','Spatial'],['seeds','Seeds'],['export','Export']];
  function renderAnalysis(){if(!analysis)return;$('analysis-progress').hidden=false;$('analysis-progress').dataset.stage=analysis.stage||'queued';$('analysis-steps').innerHTML=analysisStages.map(([key,label])=>`<li class="${analysis.steps[key]||'waiting'}"><i>${analysis.steps[key]==='done'?'✓':analysis.steps[key]==='skipped'?'−':analysis.steps[key]==='failed'?'!':''}</i>${label}</li>`).join('');$('analysis-message').textContent=analysis.message;$('analysis-time').textContent=`${Math.floor((Date.now()-analysis.started)/1000)}s`;
    const ruler=analysis.ruler;$('calibration-short').textContent=ruler?`${analysis.matrix?'Rectified scale':'Ruler'} ${fmt(analysis.matrix?analysis.pixels_per_mm:ruler.pixels_per_mm,3)} px/mm${Number.isFinite(analysis.delta_e_after)?` · ΔE00 ${fmt(analysis.delta_e_before)} → ${fmt(analysis.delta_e_after)}`:''}${analysis.qc_warning?' · Spatial QC warning':''}`:'Detecting ruler…';
    $('photo-info').textContent=analysis.steps.seeds?`${Object.values(analysis.seeds).reduce((n,v)=>n+v.length,0)} seeds measured`:'';
  }
  async function applyProgress(event,jobId){if(!analysis)return;const {seeds:eventSeeds,...fields}=event;Object.assign(analysis,fields);analysis.steps[event.stage]=event.status;if(event.completed_dish_id!=null){analysis.seeds[event.completed_dish_id]=event.seeds.map(seed=>({...seed,dish_id:event.completed_dish_id,origin:event.crop_origin}));analysis.active_dish_ids=[];}
    analysis.message=event.message || ({ruler:'Ruler scale calibrated',color:'Color correction ready',dishes:'Dishes located',spatial:event.qc_warning?'Spatial correction ready · check QC':'Spatial correction ready',seeds:'Seed masks ready',export:'Results ready'}[event.stage]);
    if(event.status==='running'&&!event.message)analysis.message=({ruler:'Detecting ruler and tick marks…',color:'Locating ColorChecker patches…',dishes:'Detecting dish outlines…',spatial:'Fitting spatial correction…',export:'Preparing measurements and CSV…'}[event.stage]||analysis.message);
    if(event.image_asset){try{const img=await imageLoad(`/api/jobs/${jobId}/asset/${event.image_asset}?revision=${event.revision}`);if(!analysis)return;analysis.frame=img;analysis.frameSize=event.image_size;}catch(err){analysis.message+=` · Preview unavailable`;}}
    if(event.dishes){$('dish-count').textContent=event.dishes.length;$('dish-list').innerHTML=event.dishes.map(d=>`<div class="progress-dish"><span>D1${d.dish_id}</span><small>Located</small></div>`).join('');}
    if(event.completed_dish_id!=null){const tiles=$('dish-list').querySelectorAll('.progress-dish');const index=analysis.dishes.findIndex(d=>d.dish_id===event.completed_dish_id);if(tiles[index])tiles[index].querySelector('small').textContent=`${event.seed_count} seeds`;}
    renderAnalysis();draw();
  }
  let lastAnimation=0;function animateAnalysis(time=0){if(!analysis)return;if(time-lastAnimation>100){lastAnimation=time;$('analysis-time').textContent=`${Math.floor((Date.now()-analysis.started)/1000)}s`;draw();}animationFrame=requestAnimationFrame(animateAnalysis);}
  function endAnalysis(){cancelAnimationFrame(animationFrame);analysis=null;$('analysis-progress').hidden=true;}
  async function runJob(request){
    endAnalysis();const original=state.image;const changedMode=state.mode!=='overview';state.result=null;state.seedKey=null;state.dishId=null;state.mode='overview';state.showPhoto=true;state.image=original;state.corrected=null;state.spatial=null;$('corrected-toggle').checked=false;$('spatial-toggle').checked=false;$('corrected-toggle').disabled=true;$('spatial-toggle').disabled=true;renderPreview();if(changedMode)fit();
    analysis={started:Date.now(),steps:{},seeds:{},active_dish_ids:[],message:'Queued for analysis…',frame:original,frameSize:[imageWidth(original),imageHeight(original)]};busy(true);$('process-button').textContent='Analyzing…';notice('');renderAnalysis();renderDisplay();animateAnalysis();
    try{const job=await request();let info,revision=0;do{info=await api(`/api/jobs/${encodeURIComponent(job.job_id)}?after_revision=${revision}`);for(const event of (info.events||[])){await applyProgress(event,job.job_id);revision=event.revision;}if(info.status==='failed')throw new Error(info.message||'Photo processing failed');if(info.status!=='done')await new Promise(r=>setTimeout(r,300));}while(info.status!=='done');
      const id=info.result_image_id||info.image_id||job.image_id;await refreshImages(id);const completed=analysis;await loadResult(id,{deferRender:true});if(state.selectedId!==id)return;watchLabels(id);busy(true);$('corrected-toggle').checked=pref('general','default_image_layer','color')==='color';$('spatial-toggle').checked=false;state.spatial=null;setSidebarView(pref('general','default_view','review'));endAnalysis();render();fit();notice('');
    }catch(err){if(analysis){analysis.steps[analysis.stage||'ruler']='failed';analysis.message=err.message;renderAnalysis();cancelAnimationFrame(animationFrame);draw();}$('process-button').textContent='Retry analysis';notice(`Analysis failed: ${err.message}`,true);}finally{busy(false);}
  }
  function drawAnalysis(){const {w,h}=dimensions();ctx.clearRect(0,0,w,h);$('canvas-shell').classList.toggle('mask-only',!state.showPhoto);ctx.save();ctx.translate(state.x,state.y);ctx.scale(state.scale,state.scale);if(state.showPhoto&&analysis.frame)ctx.drawImage(analysis.frame,0,0,...analysis.frameSize);
    const map=p=>analysis.matrix?transformPoint(p,analysis.matrix):p;
    const polygon=points=>{if(!points?.length)return false;const p=points.map(map);ctx.beginPath();ctx.moveTo(...p[0]);p.slice(1).forEach(v=>ctx.lineTo(...v));ctx.closePath();return true;};
    if($('mask-toggle').checked){ctx.lineWidth=2/state.scale;ctx.strokeStyle='#66f0d0';if(analysis.ruler){if(polygon(analysis.ruler.polygon||boxPolygon(analysis.ruler.bbox)))ctx.stroke();ctx.fillStyle='#ffbc52';for(const tick of analysis.ruler.ticks){const p=map(tick);ctx.fillRect(p[0]-2/state.scale,p[1]-2/state.scale,4/state.scale,4/state.scale);}}
      if(analysis.checker){ctx.strokeStyle='#66f0d0';if(polygon(boxPolygon(analysis.checker.bbox)))ctx.stroke();for(const center of analysis.checker.centers){const r=analysis.checker.radius;if(polygon([[center[0]-r,center[1]-r],[center[0]+r,center[1]-r],[center[0]+r,center[1]+r],[center[0]-r,center[1]+r]]))ctx.stroke();const p=map(center);ctx.fillStyle='#ffbc52';ctx.beginPath();ctx.arc(...p,3/state.scale,0,Math.PI*2);ctx.fill();}}
      for(const d of analysis.dishes||[]){const active=analysis.active_dish_ids?.includes(d.dish_id),complete=analysis.seeds[d.dish_id];ctx.strokeStyle=active?'#ffbc52':'#7ce3e7';ctx.lineWidth=(active?3:2)/state.scale;ctx.globalAlpha=active&&!matchMedia('(prefers-reduced-motion: reduce)').matches ? .65+.35*Math.sin(Date.now()/220)**2:1;ctx.setLineDash(active?[10/state.scale,5/state.scale]:[]);if(polygon(boxPolygon(d.bbox)))ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);const p=map([d.bbox[0],d.bbox[1]]);label(`D1${d.dish_id}${active?analysis.message==='Inferring seed masks'?' · Inference…':' · Measuring…':complete?` · ${complete.length} seeds`:''}`,p[0]+10/state.scale,p[1]+20/state.scale,'#123d55',true);}
      for(const seeds of Object.values(analysis.seeds)){for(const seed of seeds){ctx.fillStyle=state.showPhoto?'#ffbc5244':'#ffbc52';ctx.strokeStyle='#ffbc52';ctx.lineWidth=1.5/state.scale;for(const p of seed.polygons){if(polygon(p.map(v=>[v[0]+seed.origin[0],v[1]+seed.origin[1]]))){ctx.fill();ctx.stroke();}}if($('ids-toggle').checked&&state.scale>.3){const b=seed.bbox,p=map([b[0]+b[2]/2+seed.origin[0],b[1]+seed.origin[1]]);label(`S${String(seed.seed_id).padStart(2,'0')}`,p[0],p[1]-5/state.scale,'#172b39cc',false);}}}
    }ctx.restore();$('zoom-label').textContent=`${Math.round(state.scale*100)}%`;
  }
  function dimensions() { const rect=canvas.getBoundingClientRect(); return {w:rect.width,h:rect.height}; }
  function resize() { const {w,h}=dimensions(), dpr=window.devicePixelRatio || 1; canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); draw(); }
  function spatialActive(){return !!($('spatial-toggle').checked && state.spatial && state.result?.spatial_preview?.matrix);}
  function transformPoint(point, matrix){const [x,y]=point;const w=matrix[2][0]*x+matrix[2][1]*y+matrix[2][2];return [(matrix[0][0]*x+matrix[0][1]*y+matrix[0][2])/w,(matrix[1][0]*x+matrix[1][1]*y+matrix[1][2])/w];}
  function inverseMatrix(m){const [[a,b,c],[d,e,f],[g,h,i]]=m;const det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);if(Math.abs(det)<1e-12)return null;return [[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]].map(row=>row.map(v=>v/det));}
  function mapped(point){return spatialActive()?transformPoint(point,state.result.spatial_preview.matrix):point;}
  function boxPolygon(b){return [[b[0],b[1]],[b[0]+b[2],b[1]],[b[0]+b[2],b[1]+b[3]],[b[0],b[1]+b[3]]];}
  function mappedBox(b){const points=boxPolygon(b).map(mapped);const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);return [Math.min(...xs),Math.min(...ys),Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys)];}
  function fit() { if(!state.image)return; const {w,h}=dimensions();const box=state.mode==='dish' && selectedDish()?mappedBox(selectedDish().display_bbox||selectedDish().bbox):spatialActive()?[0,0,imageWidth(state.spatial),imageHeight(state.spatial)]:[0,0,imageWidth(state.image),imageHeight(state.image)];const pad=state.mode==='dish'?32:16;state.scale=Math.min((w-pad*2)/box[2],(h-pad*2)/box[3]);state.x=(w-box[2]*state.scale)/2-box[0]*state.scale;state.y=(h-box[3]*state.scale)/2-box[1]*state.scale;draw(); }
  function zoom(factor,cx,cy) { if(!state.image)return; const {w,h}=dimensions();cx??=w/2;cy??=h/2;const next=Math.max(.01,Math.min(8,state.scale*factor));const r=next/state.scale;state.x=cx-(cx-state.x)*r;state.y=cy-(cy-state.y)*r;state.scale=next;draw(); }
  function path(points) { if(!points?.length)return false;const transformed=points.map(mapped);ctx.beginPath();ctx.moveTo(...transformed[0]);transformed.slice(1).forEach(p=>ctx.lineTo(...p));ctx.closePath();return true; }
  function draw() {
    if(analysis){drawAnalysis();return;}
    const {w,h}=dimensions();ctx.clearRect(0,0,w,h);if(!state.image)return;
    const maskOnly=!state.showPhoto, photoOnly=!$('mask-toggle').checked;$('canvas-shell').classList.toggle('mask-only',maskOnly);ctx.save();ctx.translate(state.x,state.y);ctx.scale(state.scale,state.scale);if(state.mode==='dish'&&selectedDish()){const crop=mappedBox(selectedDish().display_bbox||selectedDish().bbox);ctx.beginPath();ctx.rect(crop[0],crop[1],crop[2],crop[3]);ctx.clip();}
    if(!maskOnly){const img=spatialActive()?state.spatial:$('corrected-toggle').checked&&state.corrected?state.corrected:state.image;ctx.drawImage(img,0,0,imageWidth(img),imageHeight(img));}
    (state.result?.dishes || []).forEach(d=>{
      if(!photoOnly && state.mode==='overview'){ctx.strokeStyle=d.dish_id===state.dishId?'#71eeec':'#7ce3e7';ctx.lineWidth=2/state.scale;ctx.setLineDash([7/state.scale,5/state.scale]);if(path(boxPolygon(d.bbox)))ctx.stroke();ctx.setLineDash([]);const p=mapped([d.bbox[0],d.bbox[1]]);label(dishName(d),p[0],p[1]-6/state.scale,'#123d55',true);}
      if(photoOnly || state.mode==='dish' && d.dish_id!==state.dishId)return;d.seeds.forEach(s=>{const selected=s.key===state.seedKey;if($('mask-toggle').checked){for(const polygon of (s.polygons || [s.polygon])){if(!path(polygon))continue;ctx.fillStyle=maskOnly?seedColor(s):seedColor(s)+'44';ctx.strokeStyle=selected?'white':seedColor(s);ctx.lineWidth=(selected?3:1.5)/state.scale;ctx.fill();ctx.stroke();}}if($('ids-toggle').checked&&(state.mode==='dish'||state.scale>.3||selected)){const b=s.bbox;const p=mapped([b[0]+b[2]/2,b[1]]);label(s.label,p[0],p[1]-5/state.scale,selected?'#123d55':'#172b39cc',false);}});
    });ctx.restore();$('zoom-label').textContent=`${Math.round(state.scale*100)}%`;
  }
  function renderDisplay(){$('mask-legend').hidden=!state.result || !$('mask-toggle').checked;for(const [id,visible,available,name]of [['photo-visibility',state.showPhoto,!!state.image,'photo'],['mask-visibility',$('mask-toggle').checked,!!state.result||!!analysis,'masks'],['ids-visibility',$('ids-toggle').checked,(!!state.result||!!analysis)&&$('mask-toggle').checked,'IDs']]){const button=$(id);button.disabled=!available;button.classList.toggle('active',visible);button.setAttribute('aria-pressed',String(visible));button.title=`${visible?'Hide':'Show'} ${name}`;button.setAttribute('aria-label',button.title);}$('ids-toggle').disabled=!$('mask-toggle').checked;}
  function label(text,x,y,bg,left) {ctx.save();const size=12/state.scale;ctx.font=`600 ${size}px Aptos, "Microsoft YaHei UI", sans-serif`;const m=ctx.measureText(text).width;const px=left?x:x-m/2;ctx.fillStyle=bg;ctx.fillRect(px-4/state.scale,y-13/state.scale,m+8/state.scale,18/state.scale);ctx.fillStyle='white';ctx.fillText(text,px,y);ctx.restore();}
  function focusDish(id) {notice(''); state.dishId=id; state.seedKey=null; state.mode='dish'; $('seed-filter').value=selectedDish()?.seeds.some(isPriority)?'ambiguous':'all';render();fit(); }
  function focusSeed(key) {notice('');const s=allSeeds().find(s=>s.key===key);if(!s)return;const changed=state.dishId!==s.dish_id||state.mode!=='dish';state.dishId=s.dish_id;state.seedKey=key;state.mode='dish';render();if(changed)fit();else draw();}
  function render() {
    const r=state.result;renderDisplay();if(!r){renderPreview();return;}$('process-button').textContent='Reanalyze';$('export-button').disabled=state.busy; $('canvas-empty').hidden=true;$('file-name').textContent=r.filename;$('backend-label').textContent=r.backend==='classical'?'Classical':'YOLO26s';$('photo-info').textContent=`${r.width} × ${r.height} px · ${r.summary.candidate_count} seeds${spatialActive()?' · spatially corrected view':''}`; $('dish-count').textContent=r.dishes.length;
    $('overview-button').hidden=state.mode==='overview';$('canvas-context').textContent=state.mode==='dish'?dishName(selectedDish()):'Overview';$('dish-mode').classList.toggle('active',state.mode==='dish');$('dish-mode').disabled=!r.dishes.length;$('focus-label').hidden=state.mode!=='dish';$('focus-label').textContent=selectedDish()?`${selectedDish().label} / Dish` : '';
    $('dish-list').innerHTML=r.dishes.map(d=>`<div class="dish-card ${d.dish_id===state.dishId?'active':''}"><button class="dish-tile" data-dish="${d.dish_id}" aria-label="Open ${escape(dishName(d))}, ${d.summary.candidate_count} seeds"><img src="${escape(d.thumbnail_url+(d.thumbnail_url.includes('?')?'&':'?')+'thumbnail=true&v='+new URL(r.corrected_url,location.origin).searchParams.get('v'))}" alt="${escape(dishName(d))} dish"></button><div class="dish-caption"><div class="dish-id-editor"><input data-sample-input="${d.dish_id}" value="${escape(state.dishDrafts?.[d.dish_id] ?? d.sample_id ?? d.sample_id_candidate ?? d.label)}" title="Edit sample ID (${escape(d.label)})" aria-label="Edit sample ID for ${escape(d.label)}" maxlength="128" ${state.busy?'disabled':''}></div><div class="dish-id-state"><span>${d.summary.candidate_count} seeds</span><span data-label-state="${d.dish_id}" role="status">${labelBadge(d)}</span></div></div></div>`).join('');
    renderPaperLabels();
    renderSeeds();renderDetail();renderCalibration();renderMeasurements();draw();
  }
  function renderPreview(){ $('overview-button').hidden=true;$('canvas-context').textContent='Overview';$('seed-detail').hidden=true; $('process-button').textContent='Analyze';const item=state.images.find(i=>i.image_id===state.selectedId);$('canvas-empty').hidden=!!state.image;$('file-name').textContent=item?.filename || 'Select a photo to begin';$('backend-label').textContent='Preview';$('photo-info').textContent=state.image?`${imageWidth(state.image)} × ${imageHeight(state.image)} px`:'—';$('dish-count').textContent='—';$('dish-list').innerHTML='';$('export-button').disabled=true;$('measurement-count').textContent='—';$('measurement-list').innerHTML='';$('read-paper-labels').disabled=true;$('paper-label-list').innerHTML='';$('review-count').textContent='—';$('seed-list').innerHTML='';$('seed-detail').innerHTML='';$('calibration-short').textContent='Not analyzed';$('calibration-detail').innerHTML='';$('dish-mode').disabled=true;$('focus-label').hidden=true;$('overview-button').classList.add('active');$('dish-mode').classList.remove('active');for(const name of ['export-seeds','export-dishes']){$(name).removeAttribute('href');$(name).setAttribute('aria-disabled','true');}draw();}
  function filteredSeeds(){const source=selectedDish()?.seeds || allSeeds();const filter=$('seed-filter').value;return source.filter(s=>filter==='all' || (filter==='ambiguous'?isPriority(s):s.review_state===filter));}
  function navigationSeeds(){const list=filteredSeeds();return list.some(s=>s.key===state.seedKey)?list:(selectedDish()?.seeds || allSeeds());}
  function renderSeeds(){const d=selectedDish(),list=filteredSeeds();$('review-title').textContent='Seeds Review';$('review-count').textContent=list.length;
    $('seed-list').innerHTML=list.length?list.map(s=>`<button class="seed-row ${s.key===state.seedKey?'active':''}" data-seed="${escape(s.key)}" aria-pressed="${s.key===state.seedKey}" title="${escape((s.ambiguity_reasons||[]).join('; '))}">${isPriority(s)?'<span class="priority-mark" aria-label="Low Confidence">▲</span>':''}<span class="seed-name">${escape(d?s.label:(state.result.dishes.find(d=>d.dish_id===s.dish_id)?.label+' / '+s.label))}</span><span class="seed-reason">${fmt(s.length_mm)} × ${fmt(s.width_mm)} mm</span><span class="state-pill ${confidenceClass(s)}">${confidenceName(s)}</span></button>`).join(''):`<div class="list-empty">${$('seed-filter').value==='ambiguous'?'No priority flags.':'No matching seeds.'}${$('seed-filter').value==='ambiguous' && (d?.seeds || allSeeds()).length?'<br><button data-show-all>View all instances</button>':''}</div>`;
  }
  function renderDetail(){const s=selectedSeed();$('seed-detail').hidden=!s;if(!s){$('seed-detail').innerHTML='';return;}const nav=navigationSeeds(),index=nav.findIndex(v=>v.key===s.key);
    const groups=[['Size',[['Length',s.length_mm,'mm'],['Width',s.width_mm,'mm'],['Projected area',s.area_mm2,'mm²']]],['Shape',[['Aspect ratio',s.aspect_ratio,''],['Circularity',s.circularity,''],['Solidity',s.solidity,'']]],['Color',[['L*',s.L,''],['a*',s.a,''],['b*',s.b,''],['Chroma C*',s.chroma,''],['Hue h°',s.hue_deg,'°']]]];
    $('seed-detail').innerHTML=`<div class="detail-heading"><h3>${escape(dishName(selectedDish()))} / ${escape(s.label)}</h3><span class="state-pill ${confidenceClass(s)}">${confidenceName(s)}</span></div><div class="review-actions"><button class="confirm" data-review="confirmed" title="Mark this instance as checked (C)" ${state.busy?'disabled':''}>Check</button><button data-review="discarded" title="Exclude (X)" ${state.busy?'disabled':''}>Exclude</button><button class="reset" data-review="pending" title="Restore the automatic assessment" ${state.busy?'disabled':''}>Restore</button></div><div class="seed-navigation"><button data-step="-1" ${state.busy||index<=0?'disabled':''}>← Previous</button><span>${index+1} / ${nav.length}</span><button data-step="1" ${state.busy||index>=nav.length-1?'disabled':''}>Next →</button></div>${groups.map(([name,metrics])=>`<details class="trait-section" ${name==='Shape'?'':'open'}><summary>${name}</summary><dl class="trait-grid">${metrics.map(([label,value,unit])=>`<div><dt>${label}</dt><dd>${fmt(value)} <small>${unit}</small></dd></div>`).join('')}</dl></details>`).join('')}<div class="color-line" title="Screen preview of corrected CIELAB D65"><span class="swatch" style="background:${safeHex(s.color_hex)}"></span>CIELAB · D65</div>${s.ambiguity_reasons.length?`<details class="seed-flags" ${isPriority(s)?'open':''}><summary>${s.ambiguity_reasons.length} review flags</summary><ul class="reason-list">${s.ambiguity_reasons.map(reason=>`<li>${escape(reason)}</li>`).join('')}</ul></details>`:''}`;
  }
  function renderPaperLabels(){document.querySelector('.paper-labels')?.classList.toggle('integration-disabled',!labelsEnabled());$('read-paper-labels').disabled=state.busy||!state.result||!labelsEnabled();$('paper-label-list').innerHTML=(state.result?.dishes||[]).filter(d=>d.paper_label?.raw_text||d.paper_label?.status).map(d=>`<div class="paper-label-row"><button type="button" class="paper-label-preview" data-paper-preview="/api/images/${state.result.image_id}/paper/${d.dish_id}" aria-label="Open paper label for ${escape(d.label)}"><img src="/api/images/${state.result.image_id}/paper/${d.dish_id}" alt="Paper label for ${escape(d.label)}" loading="lazy"></button><div><strong>${escape(d.label)}</strong><pre>${escape(d.paper_label.raw_text||'Unreadable')}</pre><small>${escape((d.paper_label.notes||[]).join(' ')||d.paper_label.error||'Machine reading; check handwriting.')}</small></div></div>`).join('');}
  function labelBadge(d){
    if(d.sample_label_status==='confirmed')return 'Saved';
    const progress=state.labelProgress[d.dish_id];
    if(progress?.status==='api_error')return '<span class="label-error">Recognition failed</span>';
    if(progress?.sample_id)return 'Auto';
    if(['queued','running'].includes(state.labelJob))return '<i class="label-spinner" aria-hidden="true"></i>Recognizing…';
    if(state.labelJob==='failed'||d.paper_label?.status==='api_error')return '<span class="label-error">Recognition failed</span>';
    return d.sample_id?'Auto':'';
  }
  function renderLabelBadges(){for(const d of state.result?.dishes||[]){const badge=$('dish-list').querySelector(`[data-label-state="${d.dish_id}"]`);if(badge)badge.innerHTML=labelBadge(d);}}
  async function watchLabels(id){const token=++labelWatchToken;try{let job;do{job=await api(`/api/results/${encodeURIComponent(id)}/labels`);if(token!==labelWatchToken||state.selectedId!==id)return;state.labelJob=job.status;for(const reading of job.labels||[]){state.labelProgress[reading.dish_id]=reading;const d=state.result?.dishes.find(d=>d.dish_id===reading.dish_id);if(d&&d.sample_label_status!=='confirmed'){d.paper_label=reading;if(reading.sample_id){d.sample_id=reading.sample_id;d.sample_id_candidate=reading.sample_id;d.sample_label_status=reading.status;const input=$('dish-list').querySelector(`[data-sample-input="${d.dish_id}"]`);if(input&&!Object.hasOwn(state.dishDrafts||{},d.dish_id)&&document.activeElement!==input)input.value=reading.sample_id;}}}renderLabelBadges();draw();$('paper-label-status').textContent=job.message||'';if(['queued','running'].includes(job.status))await new Promise(r=>setTimeout(r,700));}while(['queued','running'].includes(job.status));if(job.status==='done'){const result=await api(`/api/results/${encodeURIComponent(id)}`);if(token!==labelWatchToken||state.selectedId!==id)return;if(result.revision>=state.result.revision){state.result=result;render();}}}catch(err){if(token===labelWatchToken){state.labelJob='failed';renderLabelBadges();$('paper-label-status').textContent='Labels unavailable; enter IDs manually.';}}finally{if(token===labelWatchToken)$('read-paper-labels').disabled=state.busy||!state.result;}}
  $('read-paper-labels').onclick=async()=>{if(!state.result||state.busy)return;const id=state.selectedId;$('read-paper-labels').disabled=true;state.labelJob='queued';state.labelProgress={};renderLabelBadges();try{await api(`/api/results/${encodeURIComponent(id)}/labels`,{method:'POST'});watchLabels(id);}catch(err){state.labelJob='failed';renderLabelBadges();$('paper-label-status').textContent=err.message;$('read-paper-labels').disabled=false;}};
  $('paper-label-list').addEventListener('click',event=>{const button=event.target.closest('[data-paper-preview]');if(!button)return;const dialog=$('paper-label-lightbox'),image=$('paper-label-lightbox-image');image.src=button.dataset.paperPreview;dialog.showModal();});
  $('about-button').onclick=()=>$('about-dialog').showModal();
  $('about-close').onclick=()=>$('about-dialog').close();
  $('about-continue').onclick=()=>$('about-dialog').close();
  $('about-dialog').addEventListener('click',event=>{if(event.target===event.currentTarget)event.currentTarget.close();});
  const aboutSeenKey = 'grainmaster.about.seen.v1';
  $('about-dialog').addEventListener('close',()=>{try{localStorage.setItem(aboutSeenKey,'true');}catch{}});
  function showWelcomeAbout(){
    try{if(localStorage.getItem(aboutSeenKey)==='true')return;}catch{}
    $('about-dialog').showModal();
  }
  $('paper-label-lightbox-close').onclick=()=>$('paper-label-lightbox').close();
  $('paper-label-lightbox').addEventListener('click',event=>{if(event.target===event.currentTarget)event.currentTarget.close();});
  $('paper-label-lightbox').addEventListener('close',()=>{$('paper-label-lightbox-image').removeAttribute('src');});
  $('dish-list').addEventListener('input',event=>{const input=event.target.closest('[data-sample-input]');if(input){state.dishDrafts??={};state.dishDrafts[input.dataset.sampleInput]=input.value;}});
  async function saveSampleID(id){if(state.busy||!state.result)return;const input=$('dish-list').querySelector(`[data-sample-input="${id}"]`);const value=input.value.trim();const dish=state.result.dishes.find(d=>d.dish_id===Number(id));if(value===(dish?.sample_id??dish?.sample_id_candidate??dish?.label)){delete state.dishDrafts[id];return;}if(!value){notice('Enter a sample ID before saving.',true);return;}busy(true);try{state.result=await api(`/api/results/${encodeURIComponent(state.selectedId)}/sample-id`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dish_id:Number(id),sample_id:value})});delete state.dishDrafts[id];render();notice('Sample ID saved.');}catch(err){notice(`Save failed: ${err.message}`,true);}finally{busy(false);render();}}
  $('dish-list').addEventListener('focusout',event=>{const input=event.target.closest('[data-sample-input]');if(input&&Object.hasOwn(state.dishDrafts||{},input.dataset.sampleInput))saveSampleID(input.dataset.sampleInput);});
  $('dish-list').addEventListener('keydown',event=>{const input=event.target.closest('[data-sample-input]');if(input&&event.key==='Enter'){event.preventDefault();saveSampleID(input.dataset.sampleInput);}});
  function safeHex(value){return /^#[0-9a-f]{6}$/i.test(value||'')?value:'#ffffff';}
  function renderCalibration(){const c=state.result.calibration;$('calibration-short').textContent=`Ruler ${fmt(c.pixels_per_mm,3)} px/mm · Color ΔE00 ${fmt(c.delta_e_after)}`;$('calibration-detail').innerHTML=`<p>Spatial scale: ruler ticks, ${fmt(c.mm_per_pixel,5)} mm/px。${state.result.spatial_preview?.applied_to_measurements?'Full-image spatial correction is applied; measurements use corrected instance geometry.':'No usable spatial correction is available; measurements use the original image scale.'}</p><p>Color: ${escape(c.color_space)} · ${escape(({root_polynomial2:"second-order root polynomial",linear3:"3×3 linear matrix",affine_baseline:"affine matrix"})[c.color_model]||c.color_model||"affine matrix")}。Mean ColorChecker ΔE00：${fmt(c.delta_e_before)} → ${fmt(c.delta_e_after)}。This metric describes chart fitting; it is not an upper bound on full-image color error.</p>${state.result.spatial_preview?`<p>Spatial correction: held-out three-segment ruler scale CV ${fmt(state.result.spatial_preview.before.segment_cv_percent)}% → ${fmt(state.result.spatial_preview.after.segment_cv_percent)}%。${state.result.spatial_preview.accepted?'Internal checks passed; external validation is still required.':'Some dish geometry worsened, so not all checks passed.'}<a href="/api/images/${encodeURIComponent(state.result.image_id)}/asset/spatial_rims" target="_blank" rel="noopener">View rim detection</a>。${state.result.spatial_preview.applied_to_measurements?'Formal measurements use this correction.':'This correction is not applied to measurements for this image.'}</p>`:''}${(c.warnings||[]).map(w=>`<p>Check: ${escape(w)}</p>`).join('')}<p><a href="/api/images/${encodeURIComponent(state.result.image_id)}/asset/ruler" target="_blank" rel="noopener">View ruler calibration</a>　<a href="/api/images/${encodeURIComponent(state.result.image_id)}/asset/checker" target="_blank" rel="noopener">View ColorChecker calibration</a></p>`;}
  function measurementGroups(summary){
    return [
      ['Size',[['Count',summary.seed_count,''],['Mean length',summary.mean_length_mm,'mm'],['Mean width',summary.mean_width_mm,'mm'],['Mean projected area',summary.mean_area_mm2,'mm²']]],
      ['Shape',[['Mean aspect ratio',summary.mean_aspect_ratio,''],['Mean circularity',summary.mean_circularity,''],['Mean solidity',summary.mean_solidity,'']]],
      ['Color',[['Mean L*',summary.mean_L,''],['Mean a*',summary.mean_a,''],['Mean b*',summary.mean_b,''],['Chroma C*',summary.color_chroma,''],['Hue h°',summary.color_hue_deg,'°'],['Within-dish ΔE00 mean',summary.delta_e00_mean,''],['Within-dish ΔE00 P90',summary.delta_e00_p90,'']]]
    ];
  }
  function renderMeasurements(){
    if(!state.result){$('measurement-count').textContent='—';$('measurement-list').innerHTML='';return;}
    const r=state.result;
    $('measurement-count').textContent=r.dishes.length+' dishes';
    $('measurement-list').innerHTML=r.dishes.map(d=>{
      const summary=d.summary.draft;
      const title=r.image_id+'_'+d.dish_id;
      const sample=d.sample_id||d.sample_id_candidate;
      const groups=measurementGroups(summary).map(([name,items])=>`<section class="measurement-group"><h3>${name}</h3><dl>${items.map(([label,value,unit])=>`<div><dt>${label}</dt><dd>${fmt(value)}${unit?` <small>${unit}</small>`:''}</dd></div>`).join('')}</dl></section>`).join('');
      return `<article class="measurement-card"><header><div><strong>${escape(title)}</strong>${sample?`<small>${escape(sample)}</small>`:''}</div><span>${summary.seed_count} seeds</span></header>${groups}</article>`;
    }).join('');
  }
  function renderExportPreview(){
    if(!state.result){$('export-preview-body').innerHTML='';$('export-preview-note').textContent='—';return;}
    const mode=$('export-mode').value;
    const rows=state.result.dishes.slice(0,4).map(d=>{
      const summary=d.summary[mode];
      const sample=d.sample_id||d.sample_id_candidate||(state.result.image_id+'_'+d.dish_id);
      return `<tr><td><strong>${escape(sample)}</strong><small>${escape(d.label)}</small></td><td>${summary.seed_count}</td><td>${fmt(summary.mean_length_mm)}</td><td>${fmt(summary.mean_width_mm)}</td><td>${fmt(summary.mean_area_mm2)}</td><td>${fmt(summary.mean_aspect_ratio)}</td><td>${fmt(summary.mean_L)}</td><td>${fmt(summary.mean_a)}</td><td>${fmt(summary.mean_b)}</td></tr>`;
    }).join('');
    $('export-preview-body').innerHTML=rows;
    const extra=Math.max(0,state.result.dishes.length-4);
    $('export-preview-note').textContent=(mode==='draft'?'All retained':'Confirmed only')+(extra?` · +${extra} more dish${extra===1?'':'es'}`:'');
  }
  function updateExportLinks(){
    if(!state.result)return;
    const mode=$('export-mode').value;
    for(const kind of ['seeds','dishes']){$('export-'+kind).href='/api/results/'+encodeURIComponent(state.result.image_id)+'/export?kind='+kind+'&mode='+mode;$('export-'+kind).classList.toggle('preferred',kind===pref('export','default_kind','dishes'));}
    renderExportPreview();
  }
  function stepSeed(step){const list=navigationSeeds();if(!list.length)return;const i=list.findIndex(s=>s.key===state.seedKey);focusSeed(list[i<0?0:Math.max(0,Math.min(list.length-1,i+step))].key);$('seed-list').querySelector('.active')?.scrollIntoView({block:'nearest'});}
  async function review(value){const seed=selectedSeed();if(!seed||state.busy)return;busy(true);renderDetail();try{state.result=await api(`/api/results/${encodeURIComponent(state.result.image_id)}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dish_id:seed.dish_id,seed_id:seed.seed_id,state:value})});if($('seed-filter').value==='ambiguous'&&!(selectedDish()?.seeds||allSeeds()).some(isPriority))$('seed-filter').value='all';notice(`${selectedDish().label} / ${seed.label}: ${value==='discarded'?'Excluded':'Updated'}. Saved.`);render();if(pref('review','auto_advance',false))stepSeed(1);}catch(err){notice(`Save failed: ${err.message}`,true);}finally{busy(false);renderDetail();}}
  $('dish-list').addEventListener('click',event=>{const b=event.target.closest('[data-dish]');if(b)focusDish(Number(b.dataset.dish));});$('seed-list').addEventListener('click',event=>{const b=event.target.closest('[data-seed]');if(b)focusSeed(b.dataset.seed);if(event.target.closest('[data-show-all]')){$('seed-filter').value='all';renderSeeds();}});$('seed-detail').addEventListener('click',event=>{const b=event.target.closest('[data-review]');if(b)review(b.dataset.review);const next=event.target.closest('[data-step]');if(next&&!state.busy)stepSeed(Number(next.dataset.step));});$('seed-filter').addEventListener('change',()=>{renderSeeds();renderDetail();});
  $('overview-button').onclick=()=>{state.mode='overview';state.dishId=null;state.seedKey=null;render();fit();};$('dish-mode').onclick=()=>{if(!state.result?.dishes?.length)return;focusDish(state.dishId ?? state.result.dishes[0].dish_id);};$('fit-button').onclick=fit;$('zoom-in').onclick=()=>zoom(1.25);$('zoom-out').onclick=()=>zoom(.8);$('mask-toggle').onchange=draw;$('ids-toggle').onchange=draw;
  $('corrected-toggle').onchange=async()=>{
    if(!state.result)return;
    if($('spatial-toggle').checked){await loadSpatial();return;}
    if(!$('corrected-toggle').checked){state.displayToken++;draw();return;}
    try{
      const img=await ensureCorrected(true);
      if(!img||!$('corrected-toggle').checked)return;
      draw();
    }catch(err){
      $('corrected-toggle').checked=false;
      notice(err.message,true);
      draw();
    }
  };
  async function loadSpatial(){if(!state.result?.spatial_preview)return;const token=state.loadToken,displayToken=++state.displayToken;try{const img=await imageLoad($('corrected-toggle').checked?state.result.spatial_preview.color_url:state.result.spatial_preview.url);if(token!==state.loadToken||displayToken!==state.displayToken||!$('spatial-toggle').checked)return;state.spatial=img;render();}catch(err){if(token!==state.loadToken||displayToken!==state.displayToken)return;$('spatial-toggle').checked=false;notice(err.message,true);draw();}}
  $('spatial-toggle').onchange=async()=>{if($('spatial-toggle').checked)await loadSpatial();else{state.displayToken++;render();}};
  $('photo-visibility').onclick=()=>{state.showPhoto=!state.showPhoto;renderDisplay();draw();};$('mask-visibility').onclick=()=>{$('mask-toggle').checked=!$('mask-toggle').checked;renderDisplay();draw();};$('ids-visibility').onclick=()=>{if($('ids-visibility').disabled)return;$('ids-toggle').checked=!$('ids-toggle').checked;renderDisplay();draw();};
  function setSidebarView(view){state.sidebarView=view;const measurements=view==='measurements';$('review-sidebar').hidden=measurements;$('measurement-sidebar').hidden=!measurements;$('tab-inspect').classList.toggle('active',!measurements);$('tab-summary').classList.toggle('active',measurements);$('tab-inspect').setAttribute('aria-selected',String(!measurements));$('tab-summary').setAttribute('aria-selected',String(measurements));if(measurements)renderMeasurements();}
  $('tab-inspect').onclick=()=>setSidebarView('review');$('tab-summary').onclick=()=>setSidebarView('measurements');$('export-button').onclick=()=>{if(!state.result)return;$('export-mode').value=pref('export','default_mode','draft');updateExportLinks();$('export-dialog').showModal();};$('export-close').onclick=()=>$('export-dialog').close();$('export-dialog').addEventListener('click',event=>{if(event.target===event.currentTarget)event.currentTarget.close();});$('export-mode').onchange=updateExportLinks;
  document.addEventListener('grainmaster:settings-changed',()=>{if(!state.result)return;$('read-paper-labels').disabled=state.busy||!labelsEnabled();document.querySelector('.paper-labels')?.classList.toggle('integration-disabled',!labelsEnabled());});
  $('photo-search').oninput=renderPhotos;
  $('image-select').onchange=()=>selectImage($('image-select').value);$('photo-list').addEventListener('click',event=>{const remove=event.target.closest('[data-delete-image]');if(remove){deleteImage(remove.dataset.deleteImage);return;}const button=event.target.closest('[data-image]');if(button&&!state.busy)selectImage(button.dataset.image);});$('process-button').onclick=()=>{const id=state.selectedId;if(!id||state.busy)return;const force=!!state.result||!!analysis;runJob(()=>api('/api/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_id:id,force})}));};$('upload-button').onclick=()=>$('upload').click();$('folder-button').onclick=()=>$('folder-upload').click();$('upload').onchange=()=>importFiles($('upload').files);$('folder-upload').onchange=()=>importFiles($('folder-upload').files);
  canvas.addEventListener('wheel',event=>{if(!state.image)return;event.preventDefault();const b=canvas.getBoundingClientRect();zoom(Math.exp(-event.deltaY*.0015),event.clientX-b.left,event.clientY-b.top);},{passive:false});
  canvas.addEventListener('pointerdown',event=>{if(!state.image || event.button!==0)return;state.drag={startX:event.clientX,startY:event.clientY,x:state.x,y:state.y,moved:false};canvas.setPointerCapture(event.pointerId);canvas.classList.add('dragging');});
  canvas.addEventListener('pointermove',event=>{if(!state.drag)return;const dx=event.clientX-state.drag.startX,dy=event.clientY-state.drag.startY;if(Math.hypot(dx,dy)>4)state.drag.moved=true;state.x=state.drag.x+dx;state.y=state.drag.y+dy;draw();});
  canvas.addEventListener('pointerup',event=>{if(!state.drag)return;const moved=state.drag.moved;state.drag=null;canvas.classList.remove('dragging');if(moved||!state.result)return;const rect=canvas.getBoundingClientRect(),px=(event.clientX-rect.left-state.x)/state.scale,py=(event.clientY-rect.top-state.y)/state.scale;let original=[px,py];if(spatialActive()){const inverse=inverseMatrix(state.result.spatial_preview.matrix);if(!inverse)return;original=transformPoint(original,inverse);}const [ox,oy]=original;const seeds=allSeeds().filter(s=>state.mode==='overview'||s.dish_id===state.dishId);const seed=[...seeds].reverse().find(s=>(s.polygons || [s.polygon]).some(p=>insidePolygon(ox,oy,p)));if(seed)focusSeed(seed.key);else if(state.mode==='overview'){const dish=state.result.dishes.find(d=>ox>=d.bbox[0]&&oy>=d.bbox[1]&&ox<=d.bbox[0]+d.bbox[2]&&oy<=d.bbox[1]+d.bbox[3]);if(dish)focusDish(dish.dish_id);}});
  canvas.addEventListener('pointercancel',()=>{state.drag=null;canvas.classList.remove('dragging');});canvas.addEventListener('keydown',event=>{if(event.key==='+'||event.key==='=')zoom(1.25);else if(event.key==='-')zoom(.8);else if(event.key==='0')fit();else if(event.key==='Escape')$('overview-button').click();else if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();state.x+=event.key==='ArrowLeft'?35:event.key==='ArrowRight'?-35:0;state.y+=event.key==='ArrowUp'?35:event.key==='ArrowDown'?-35:0;draw();}});
  function insidePolygon(x,y,points){if(!points?.length)return false;let inside=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])inside=!inside;}return inside;}
  document.addEventListener('keydown',event=>{if(event.ctrlKey||event.metaKey||event.altKey||state.busy||$('inspect-view').hidden||/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName))return;const key=event.key.toLowerCase();if(key==='c'&&selectedSeed()){event.preventDefault();review('confirmed');}else if(key==='x'&&selectedSeed()){event.preventDefault();review('discarded');}else if(key==='n'){event.preventDefault();stepSeed(1);}else if(key==='p'){event.preventDefault();stepSeed(-1);}});
  let resizeTimer, lastSize=dimensions();new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if($('inspect-view').hidden)return;const size=dimensions(), widthChanged=Math.abs(size.w-lastSize.w)>80;state.x+=(size.w-lastSize.w)/2;state.y+=(size.h-lastSize.h)/2;lastSize=size;resize();if(widthChanged && state.image)fit();},100);}).observe($('canvas-shell'));
  async function init(){resize();try{await prefs().init();await refreshImages();let remembered='';if(pref('general','remember_last_photo',true))try{remembered=localStorage.getItem('grainmaster:last-photo')||'';}catch{}const item=state.images.find(i=>i.image_id===remembered)||state.images.find(i=>i.processed&&!i.is_example)||state.images.find(i=>i.processed)||state.images.find(i=>i.is_example)||state.images[0];if(item){$('image-select').value=item.image_id;if(item.processed)await loadResult(item.image_id);else await selectImage(item.image_id);}else notice('Import a photo, then click “Analyze”.');}catch(err){notice(`Could not connect to the workbench service: ${err.message}. Confirm that the local service is running.`,true);}}showWelcomeAbout();init();
})();
