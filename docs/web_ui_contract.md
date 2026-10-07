# Web UI P0 shared contract

Local FastAPI + static vanilla JS canvas. Bind127.0.0.1:8765. Review saved under artifacts/web_ui (neverraw). Root owns server.py launcher/pyproject/docs. Frontend owns web/*. Backend review agent owns web_review.py/tests/test_web_review.py. API agent owns tests/test_web_server.py.

GET /api/images -> {images:[{image_id,filename,processed,dish_count?,seed_count?}]}
POST /api/process JSON {image_id} -> {job_id}; POST /api/upload multipart field file -> {job_id,image_id}. Upload new immutable copy to data/processed/web_uploads/<uuid>.jpg preserving raw. Originalfilename metadata.
GET /api/jobs/<job_id> -> {job_id,image_id,status:queued|running|done|failed,message,result_image_id?}. Background singleworker. No fakeprogress percentage.
GET /api/results/<image_id> -> result below. Cached0462 immediatelyavailable without rerun. All IDs simplealphanumericdashunderscore and only registeredpaths.
GET /api/images/<id>/original -> imagebytes; GET /api/images/<id>/asset/<name> whitelist corrected,ruler,checker,preview; GET /api/images/<id>/dish/<dish_id>?corrected=true -> imagecrop JPEG.
POST /api/results/<id>/review JSON {dish_id:int,seed_id:int,state:confirmed|discarded|pending} -> refreshedresult. Persistenceatomicreview.json, measurements/masks unchanged, statependingundoallowed. No bulkconfirmationbydefault.
GET /api/results/<id>/export?kind=seeds|dishes&mode=confirmed|draft -> utf8bomCSV. Confirmed excludespending/discarded; draft excludesdiscarded butincludespending with reviewstate/provenance. Mean traits/colors recalculated on current includedinstances. Discard nevercreatesnewmask or GT.

Result JSON:
{image_id,filename,width,height,backend,revision?,image_url,corrected_url,
 calibration:{pixels_per_mm,mm_per_pixel,scale_source:'long_ruler',color_space:'CIELAB D65',delta_e_before,delta_e_after,warnings:[string]},
 dishes:[{dish_id,label:'D11'for1 /'D12'for2 etc (displayonly),bbox:[x,y,w,h],center:[x,y],thumbnail_url,
  seeds:[{dish_id,seed_id,key:'1:1',label:'S01',bbox:[x,y,w,h] ORIGINALcoords,polygon:[[x,y],...] ORIGINALcoords,
          length_mm,width_mm,area_mm2,L,a,b,seg_confidence,qc_flag,review_state:'pending',
          ambiguity_reasons:[Chinese string],color_hex:'#rrggbb'}],
  summary:{candidate_count:non-discarded,confirmed_count,pending_count,discarded_count,
     draft:{seed_count,mean_length_mm,mean_width_mm,mean_area_mm2,mean_L,mean_a,mean_b,color_hex},
     confirmed:{samekeys,count0meansnull}}}],
 summary:{candidate_count,confirmed_count,pending_count,discarded_count},
 color_method:'每粒种子为校色后mask内部中位数，培养皿为种子间等权平均Lab（D65）'}

review module API:
load_result(output_dir:Path,*,image_id:str,filename:str,review_path:Path,image_url:str,corrected_url:str)->dict
save_review(review_path:Path,result:dict,dish_id:int,seed_id:int,state:str)->None; validateinstanceIDs andstate.
export_csv(result:dict,kind:str='seeds',mode:str='confirmed')->str (withBOMorserveraddBOMspecify)
Review reasons: classicalcandidateflag separate from specific shapeheuristics; flag large/small area relative dishmedian, low solidity, fragmented/touching borders, confidence onlyYOLO learned meaningful. No falseprobability. All classical seeds pendinginitial; highlight specificambiguityreasons mosturgentreview. Important neitherconfirmation norcolorswatch certifiesgroundtruth/calibrationaccuracy.

Frontend: Chinese. App canvas hero rather thanmarketingpage; spatial IDs right thumbnailsclickzoom tocorrespondingdish, overviewreturn, dragpan,wheelzoom,fit,original/correctedtoggle,mask andIDtoggle. Clickseed polygon select ->detail+confirm/discard/pendingundo; ambiguityfilter queue. Color quantitativeLab3components notsingle arbitraryrank; averagecolor swatch illustrative. Summary tab with draftvsconfirmed toggle anddownloadlinks. Upload/chooseexisting0462,error/loadingfeedback,retainselectionafterreviewrefresh. Mobilewrap. No fake data.

Design: ink#233344, navy#123d55, coolwhite#f7f9fb, cyan#148c9b,blue#265dcd,amber#cc8424. LocalChineseMicrosoftYaHeiUI, EnglishAptos. Photo dominates; quiet toolbarandnavybrandrail. Visible keyboardfocus and accessiblecontrols, responsivepanels.
