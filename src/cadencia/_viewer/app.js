"use strict";
const $ = id => document.getElementById(id);
const audio = $("audio");
const state = {manifest:null,song:null,chart:null,features:null,start:0,span:12,selected:null,songRequest:0,chartRequest:0,height:650,pendingSeek:null,include:[]};
const palette = ["#99d9ca","#eaa2bd","#b8b0e5","#e5c189","#8ac9e0","#d0d88f","#e5aa8d","#9fc5a1","#d6a9d8"];
const colors = {muted:"#a4b2b9",amber:"#e5c189",line:"#30393e"};
const esc = value => String(value ?? "—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pathURL = path => "/data/"+path.split("/").map(encodeURIComponent).join("/");
const fmt = (n,d=3) => n == null ? "—" : Number(n).toFixed(d);
const columnColor = n => palette[n%palette.length];
const contextEvent = event => !["hit","hold"].includes(event.category);
function clock(t){t=Math.max(0,t||0);return `${Math.floor(t/60)}:${(t%60).toFixed(3).padStart(6,"0")}`;}
function status(message,error=false){$("status").textContent=message;$("status").className=error?"error":"";}
async function getJSON(url){const response=await fetch(url);if(!response.ok)throw new Error(`${response.status}: ${url}. Load a chart with cadencia.osumania.load_chart(), then refresh.`);return response.json();}
function setWindow(t){if(!state.song)return;state.start=Math.max(0,Math.min(t,Math.max(0,state.song.duration_s-state.span)));$("position").value=state.start;render();}
function seek(t){if(!state.song)return;const target=Math.max(0,Math.min(Number(t)||0,state.song.duration_s));if(audio.readyState===0){state.pendingSeek=target;return;}audio.currentTime=target;$("seek").value=fmt(target,2);updateCursor();}
function svgLine(x1,y1,x2,y2,color,width=1,extra=""){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`;}
function text(x,y,label,color=colors.muted,extra=""){return `<text x="${x}" y="${y}" fill="${color}" ${extra}>${esc(label)}</text>`;}
function setSourceLink(record){let url=record.source_url;if(!url&&Number(record.beatmap_id)>0)url=`https://osu.ppy.sh/beatmaps/${record.beatmap_id}`;$("detail-link").hidden=!/^https?:\/\//i.test(url||"");if(!$("detail-link").hidden)$("detail-link").href=url;}

function decodeTrace(trace){
  const source=new Map(trace.raw_events.map(event=>[event.id,event]));
  const counts=new Map();
  Object.values(trace.lanes).flat().forEach(event=>counts.set(event.start_s,(counts.get(event.start_s)||0)+1));
  const notes=Object.values(trace.lanes).flat().map(event=>({
    ...event,event_id:event.id,column:event.lane,action_type:event.kind,source_variant:event.kind,
    type_flags:event.source_type,end_s:event.end_s??null,duration_s:event.kind==="hold"?event.end_s-event.start_s:null,
    simultaneous_count:counts.get(event.start_s),beat_position:source.get(event.id)?.beat_position,
    source_text:source.get(event.id)?.source_text,conversion_status:"decoded",
  })).sort((a,b)=>a.start_s-b.start_s||a.column-b.column||a.source_line-b.source_line);
  const frames=trace.onset_strength.times.map((time,i)=>({
    time_s:time,onset_strength:trace.onset_strength.values[i],rms:trace.rms_energy.values[i],
    spectral_centroid_hz:trace.spectral_centroid.values[i],harmonic_fraction:trace.harmonic_energy_fraction.values[i],
    waveform_min:trace.waveform.minimum[i]??null,waveform_max:trace.waveform.maximum[i]??null,
  }));
  return {
    chart:{...trace,notes,column_count:4,chart_hash:trace.metadata.chart_sha256,
      difficulty_label:trace.metadata.difficulty,charter:trace.metadata.charter,time_origin:trace.metadata.time_origin,
      coverage_status:trace.metadata.coverage_status,
      raw_events:trace.raw_events.map(event=>({...event,event_id:event.id})),
      diagnostics:trace.diagnostics.map(d=>({...d,severity:d.code.startsWith("invalid_")?"error":"warning"})),
    },
    features:{frames,chroma:trace.chroma,
      onsets:trace.candidate_onsets.times.map((time,i)=>({time_s:time,strength:trace.candidate_onsets.values[i]})),
      estimated_beats:trace.estimated_beats.map(time=>({time_s:time})),
      metadata:{analysis_sample_rate:trace.metadata.analysis.sample_rate},
    },
  };
}

async function loadSong(id){
  const request=++state.songRequest;audio.pause();state.pendingSeek=null;state.chart=null;state.features=null;state.selected=null;state.start=0;
  $("play").disabled=true;$("chart").disabled=true;$("trace").innerHTML="";$("event-table").innerHTML="";
  status("Loading original audio and Python feature exports…");
  state.song=state.manifest.songs.find(s=>s.song_id===id);const song=state.song;
  $("chart").innerHTML=song.charts.map(c=>`<option value="${esc(c.chart_id)}">${c.column_count}K · ${esc(c.difficulty_label)}</option>`).join("");
  $("song-meta").textContent=`${song.artist} · ${song.charts.length} chart${song.charts.length===1?"":"s"}`;
  $("duration").textContent=`${clock(song.duration_s)} / source audio`;
  $("position").max=Math.max(0,song.duration_s-state.span);$("position").value=0;$("seek").max=song.duration_s;$("seek").value=0;
  audio.src=pathURL(song.audio_path);audio.load();
  try{
    $("chart").disabled=false;
    const query=new URLSearchParams(location.search);
    const desired=(query.get("song")===id?song.charts.find(c=>c.chart_id===query.get("chart")):null)||song.charts[0];
    $("chart").value=desired.chart_id;
    await loadChart(desired.chart_id);
    if(request!==state.songRequest)return;
    const at=Number(query.get("t"));if(query.get("song")===id&&Number.isFinite(at)&&at>0){seek(at);setWindow(at-2);}
  }catch(error){if(request===state.songRequest){status(`Unable to load recording. ${error.message}`,true);$("coverage").textContent="Data unavailable. Missing layers are not measured silence.";}}
}

async function loadChart(id){
  const request=++state.chartRequest,songRequest=state.songRequest;
  const song=state.song,record=song.charts.find(c=>c.chart_id===id);
  state.chart=null;state.selected=null;$("play").disabled=true;$("trace").innerHTML="";$("event-table").innerHTML="";$("inspection").textContent="Loading source events…";
  $("coverage").textContent="Loading chart coverage…";status("Loading chart events and timing points…");
  try{
    const trace=await getJSON(pathURL(record.processed_path));if(request!==state.chartRequest||songRequest!==state.songRequest)return;
    const {chart,features}=decodeTrace(trace);
    state.chart=chart;state.features=features;setSourceLink(record);
    state.featureMax=features.frames.reduce((m,f)=>({onset:Math.max(m.onset,f.onset_strength),rms:Math.max(m.rms,f.rms)}),{onset:.01,rms:.01});
    $("stat-starts").textContent=chart.notes.length.toLocaleString();
    $("stat-holds").textContent=chart.notes.filter(n=>n.action_type==="hold"&&n.duration_s!=null).length;
    $("stat-columns").textContent=`${chart.column_count}K`;
    const tempos=chart.timing_map.map(t=>t.bpm);
    $("stat-bpm").textContent=tempos.length?`${fmt(Math.min(...tempos),0)}${Math.max(...tempos)!==Math.min(...tempos)?"–"+fmt(Math.max(...tempos),0):""} BPM`:"No grid";
    const invalid=chart.notes.filter(n=>!n.analysis_eligible).length,errors=chart.diagnostics.filter(d=>d.severity==="error").length;
    const complete=chart.coverage_status==="complete";
    $("coverage").classList.toggle("complete",complete&&!invalid&&!errors);
    $("coverage").textContent=`${complete?"HIT/HOLD DECODE COMPLETE":"PARTIAL HIT/HOLD DECODE"} · ${chart.notes.length} source events · ${invalid} ineligible · ${errors} conversion errors. Original lanes and simultaneous notes remain separate.`;
    $("column-legend").innerHTML=Array.from({length:4},(_,i)=>i).filter(i=>state.include.includes(`lane${i}`)).map(i=>`<span style="color:${columnColor(i)}">● lane${i}</span>`).join("");
    $("anchor").textContent=`${chart.time_origin}. Beat zero: ${chart.timing_map.length?fmt(chart.timing_map[0].start_s)+" s":"unavailable"}.`;
    $("diagnostics").innerHTML=chart.diagnostics.length?chart.diagnostics.map(d=>`<p><strong>${esc(d.code)}</strong>${d.source_line?" · L"+d.source_line:""}<br>${esc(d.message)}</p>`).join(""):"No parser diagnostics.";
    const base=record.processed_path.replace(/trace\.json$/,"");
    const links=[["Complete trace.json",record.processed_path],["Original .osu",base+"chart.osu"],["Original archive",base+"archive.osz"],["Download provenance",base+"download.json"]];
    $("downloads").innerHTML=links.map(([name,path])=>`<a href="${pathURL(path)}" target="_blank">${esc(name)} ↗</a>`).join("");
    renderInspection();render();$("play").disabled=false;
    status(`${song.title} · ${chart.column_count}K · ${chart.difficulty_label} · ${chart.charter} · ${chart.notes.length} events loaded`);
  }catch(error){if(request===state.chartRequest){status(`Unable to load chart. ${error.message}`,true);$("coverage").textContent="Chart unavailable. Check trace.json and the cached song path.";}}
}

function render(){
  const {song,chart,features,start,span}=state;if(!song||!chart||!features)return;
  const focused=document.activeElement?.dataset?.event;
  $("window-label").textContent=`${fmt(start,1)}–${fmt(start+span,1)} s`;
  const left=125,right=1080,width=right-left,x=t=>left+(t-start)/span*width;
  const positions={},rows=[];let nextY=44;
  function addRow(name,key,height,color=colors.muted){positions[key]=nextY+height/2;rows.push([name,positions[key],color,height]);nextY+=height+8;}
  for(let lane=0;lane<4;lane++)if(state.include.includes(`lane${lane}`))addRow(`LANE ${lane}`,`lane${lane}`,40,columnColor(lane));
  if(state.include.includes("waveform"))addRow("WAVEFORM","wave",64);
  if(state.include.includes("onset_strength")||state.include.includes("candidate_onsets"))addRow("ONSETS","onset",64,colors.amber);
  if(state.include.includes("rms_energy"))addRow("RMS ENERGY","rms",64,palette[0]);
  if(state.include.includes("spectral_centroid"))addRow("CENTROID","centroid",64,palette[2]);
  if(state.include.includes("harmonic_energy_fraction"))addRow("HARMONIC","harmonic",64,palette[2]);
  if(state.include.includes("chroma"))addRow("CHROMA","chroma",132,palette[0]);
  if(state.include.includes("estimated_beats"))addRow("EST. BEATS","beats",40,palette[4]);
  if(state.include.includes("timing"))addRow("CONTEXT","context",40,colors.amber);
  const height=Math.max(140,nextY+20);state.height=height;$("trace").setAttribute("viewBox",`0 0 1100 ${height}`);
  let svg=`<defs><clipPath id="plot-clip"><rect x="${left}" y="24" width="${width}" height="${height-42}"/></clipPath></defs>`;
  rows.forEach(([name,y,c,h])=>{svg+=`<rect x="${left}" y="${y-h/2}" width="${width}" height="${h}" fill="#131b1e"/>`;svg+=text(2,y-3,name,c);svg+=svgLine(left,y+h/2,right,y+h/2,colors.line,.6);});
  for(const [key,label] of [["wave","amplitude ±1"],["onset","spectral flux"],["rms","linear amplitude"],["centroid","Hz"],["harmonic","HPSS ratio 0–1"],["chroma","pitch classes"]])if(positions[key]!=null)svg+=text(2,positions[key]+15,label);
  if(!rows.length)svg+=text(left+8,70,"No timeline layers selected.");
  const step=span<=8?1:span<=24?2:5;
  for(let t=Math.ceil(start/step)*step;t<=start+span;t+=step)svg+=text(x(t),17,`${t.toFixed(0)} s`,colors.muted,'text-anchor="middle"');
  if(positions.chroma!=null)features.chroma.labels.forEach((label,pitch)=>{svg+=text(100,positions.chroma-52+(11-pitch)*10,label);});
  svg+=`<g clip-path="url(#plot-clip)">`;
  chart.beat_grid.forEach(t=>{if(t.time_s<start||t.time_s>start+span)return;
    if($("grid").checked)svg+=svgLine(x(t.time_s),24,x(t.time_s),height-20,"#3b484f",.7,'opacity=".65"');
    if($("measures").checked&&t.is_barline){svg+=svgLine(x(t.time_s),24,x(t.time_s),height-20,"#8c9ca3",1.5);svg+=text(x(t.time_s)+3,37,`bar (${t.meter})`,colors.muted);}
  });
  if(positions.beats!=null&&$("estimated").checked)features.estimated_beats.forEach(b=>{if(b.time_s>=start&&b.time_s<=start+span){svg+=svgLine(x(b.time_s),24,x(b.time_s),height-20,"#6da0a1",1,'stroke-dasharray="2 6" opacity=".45"');svg+=svgLine(x(b.time_s),positions.beats-9,x(b.time_s),positions.beats+9,palette[4],2);}});
  const frames=features.frames.filter(f=>f.time_s>=start-.1&&f.time_s<=start+span+.1);
  function tracePath(key,y,scale,c){let d="",active=false;frames.forEach(f=>{if(f[key]==null){active=false;return;}d+=`${active?"L":"M"}${x(f.time_s).toFixed(2)},${(y-Math.min(1,Math.max(0,f[key]/scale))*45).toFixed(2)} `;active=true;});svg+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="1.5"/>`;}
  if(positions.wave!=null){let wave="";frames.forEach(f=>{if(f.waveform_min!=null)wave+=`M${x(f.time_s).toFixed(2)},${(positions.wave-f.waveform_min*27).toFixed(2)}V${(positions.wave-f.waveform_max*27).toFixed(2)} `;});svg+=`<path d="${wave}" stroke="#75868e" stroke-width="1"/>`;}
  if(state.include.includes("onset_strength"))tracePath("onset_strength",positions.onset+22,state.featureMax.onset,colors.amber);
  if(positions.rms!=null)tracePath("rms",positions.rms+22,state.featureMax.rms,palette[0]);
  if(positions.centroid!=null)tracePath("spectral_centroid_hz",positions.centroid+22,features.metadata.analysis_sample_rate/2,palette[2]);
  if(positions.harmonic!=null)tracePath("harmonic_fraction",positions.harmonic+22,1,palette[2]);
  if(positions.onset!=null)svg+=text(right-4,positions.onset-26,`max ${fmt(state.featureMax.onset,1)}`,colors.muted,'text-anchor="end"');
  if(positions.rms!=null)svg+=text(right-4,positions.rms-26,`max ${fmt(state.featureMax.rms)} FS`,colors.muted,'text-anchor="end"');
  if(positions.centroid!=null)svg+=text(right-4,positions.centroid-26,`0–${features.metadata.analysis_sample_rate/2} Hz`,colors.muted,'text-anchor="end"');
  if(state.include.includes("candidate_onsets")&&$("candidates").checked)features.onsets.forEach(o=>{if(o.time_s>=start&&o.time_s<=start+span){svg+=svgLine(x(o.time_s),positions.onset-20,x(o.time_s),positions.onset+27,colors.amber,1,'opacity=".4"');svg+=`<circle cx="${x(o.time_s)}" cy="${positions.onset-20}" r="2" fill="${colors.amber}"/>`;}});
  if(positions.chroma!=null){
    const series=features.chroma,hop=series.times.length>1?series.times[1]-series.times[0]:.02;
    series.times.forEach((time,i)=>{if(time<start-hop||time>start+span)return;series.values[i].forEach((value,pitch)=>{svg+=`<rect x="${x(time)}" y="${positions.chroma-60+(11-pitch)*10}" width="${Math.max(.5,hop/span*width+.2)}" height="10" fill="hsl(165 45% ${10+Math.min(1,value)*55}%)"/>`;});});
  }
  const visible=chart.notes.filter(n=>positions[`lane${n.column}`]!=null&&n.start_s<=start+span&&Math.max(n.end_s??n.start_s,n.start_s)>=start);
  visible.forEach(n=>{
    const y=positions[`lane${n.column}`],c=columnColor(n.column),selected=state.selected===n.event_id?" selected":"";
    const label=`${n.action_type} lane${n.column} at ${fmt(n.start_s)} seconds`;
    svg+=`<g class="note${selected}" role="button" tabindex="0" data-event="${esc(n.event_id)}" aria-label="${esc(label)}" fill="${c}"><title>${esc(label)}</title>`;
    if(!n.analysis_eligible){svg+=`<path d="M${x(n.start_s)},${y-8}l8,8 -8,8 -8,-8Z" fill="none" stroke="${c}" stroke-width="2"/>`;}
    else if(n.action_type==="hold"&&n.duration_s!=null){svg+=`<rect x="${x(n.start_s)}" y="${y-7}" width="${Math.max(1,x(n.end_s)-x(n.start_s))}" height="14" rx="4" fill="${c}" fill-opacity=".24" stroke="${c}"/>`;svg+=svgLine(x(n.start_s),y-12,x(n.start_s),y+12,c,3)+svgLine(x(n.end_s),y-10,x(n.end_s),y+10,c,2);}
    else svg+=`<circle cx="${x(n.start_s)}" cy="${y}" r="5.5" stroke="${c}"/>`;
    svg+="</g>";
  });
  if(positions.context!=null&&$("context").checked){chart.raw_events.filter(contextEvent).forEach(e=>{if(e.start_s==null||e.start_s>start+span||(e.end_s??e.start_s)<start)return;
    const label=`${e.category} at ${fmt(e.start_s)} seconds`;
    svg+=`<rect class="note" role="button" tabindex="0" data-event="${esc(e.event_id)}" aria-label="${esc(label)}" x="${x(e.start_s)-2}" y="${positions.context-9}" width="${e.end_s?Math.max(4,x(e.end_s)-x(e.start_s)):4}" height="18" fill="${e.category==="tempo"?palette[0]:colors.amber}" fill-opacity=".65"><title>${esc(label)}</title></rect>`;
  });}else if(positions.context!=null)svg+=text(left+8,positions.context,"Timing/context hidden; original records remain available.",colors.amber);
  svg+=`<line id="cursor" x1="0" x2="0" y1="24" y2="${height-20}" stroke="#f6f5e8" stroke-width="1.3" pointer-events="none"/></g>`;
  $("trace").innerHTML=svg;
  $("visible-count").textContent=`${visible.length} events · ${fmt(start,1)}–${fmt(start+span,1)} s`;
  $("event-table").innerHTML=visible.length?visible.map(n=>`<tr><td><button data-event="${esc(n.event_id)}" aria-label="Inspect ${esc(n.action_type)} lane${n.column} at ${fmt(n.start_s)} seconds">Inspect</button></td><td>${fmt(n.start_s)}</td><td>lane${n.column}</td><td>${esc(n.action_type)}${n.simultaneous_count>1?` · ${n.simultaneous_count} simultaneous`:""}${!n.analysis_eligible?" · invalid":""}</td><td>${fmt(n.duration_s)}</td><td>type ${n.type_flags} · L${n.source_line}</td></tr>`).join(""):"<tr><td colspan='6'>No displayed hit/hold event in this window. Check the selected layers and chart coverage.</td></tr>";
  renderOverview();updateCursor();
  if(focused){const replacement=Array.from($("trace").querySelectorAll("[data-event]")).find(e=>e.dataset.event===focused);replacement?.focus({preventScroll:true});}
}

function renderOverview(){if(!state.features)return;$("overview").hidden=!state.include.includes("waveform");const {song,features,start,span}=state;let d="";
  features.frames.forEach((f,i)=>{if(i%5||f.waveform_min==null)return;const x=f.time_s/song.duration_s*1100;d+=`M${x.toFixed(1)},${40-f.waveform_min*32}V${40-f.waveform_max*32} `;});
  $("overview").innerHTML=`<path d="${d}" stroke="#62767b"/><rect x="${start/song.duration_s*1100}" y="1" width="${Math.min(span,song.duration_s)/song.duration_s*1100}" height="78" fill="#99d9ca" fill-opacity=".13" stroke="#99d9ca"/><line id="overview-cursor" x1="0" x2="0" y1="0" y2="80" stroke="#f6f5e8"/>`;
}
function selectEvent(id,move=false){state.selected=id;renderInspection();const event=state.chart.notes.find(n=>n.event_id===id)||state.chart.raw_events.find(e=>e.event_id===id);if(move&&event?.start_s!=null){seek(event.start_s);setWindow(event.start_s-state.span/4);}else render();}
function renderInspection(){if(!state.chart)return;const n=state.chart.notes.find(n=>n.event_id===state.selected)||state.chart.raw_events.find(e=>e.event_id===state.selected);
  if(!n){$("inspect-title").textContent="Select an event";$("inspection").textContent="No event selected.";return;}
  $("inspect-title").textContent=n.action_type||n.category;
  const fields=[["Audio start",n.start_s==null?"unavailable":`${fmt(n.start_s)} s`],["Audio end",n.end_s==null?"—":`${fmt(n.end_s)} s`],["Duration",n.duration_s==null?"—":`${fmt(n.duration_s)} s`],["Beat position",fmt(n.beat_position,4)],["Lane",n.column==null?"—":`lane${n.column}`],["Simultaneous",n.simultaneous_count??"—"],["Source variant",n.source_variant||n.category],["Flags / source x",n.type_flags==null?"—":`${n.type_flags} / ${n.source_x}`],["Source line",n.source_line],["Conversion",n.conversion_status],["Analysis eligible",n.analysis_eligible===undefined?"context":String(n.analysis_eligible)]];
  if(n.category==="tempo")fields.push(["Beat length",`${fmt(n.beat_length_ms)} ms`],["Meter",n.meter]);
  if(n.category==="scroll_control")fields.push(["Scroll multiplier",fmt(n.scroll_multiplier)]);
  $("inspection").innerHTML="<dl>"+fields.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")+`</dl><pre>${esc(n.source_text)}\n\n${esc(n.event_id)}\nchart SHA256: ${esc(state.chart.chart_hash)}</pre>`;
}
function updateCursor(){const t=audio.currentTime;$("clock").textContent=clock(t);$("cursor-label").textContent=`Cursor: ${fmt(t)} s`;
  const line=$("cursor");if(line){const x=125+(t-state.start)/state.span*955;line.setAttribute("x1",x);line.setAttribute("x2",x);}
  const overview=$("overview-cursor");if(overview&&state.song){const x=t/state.song.duration_s*1100;overview.setAttribute("x1",x);overview.setAttribute("x2",x);}
}
function frame(){if(!audio.paused&&state.chart&&$("follow").checked&&(audio.currentTime<state.start||audio.currentTime>state.start+state.span*.9))setWindow(audio.currentTime-state.span*.15);updateCursor();requestAnimationFrame(frame);}
$("song").addEventListener("change",e=>loadSong(e.target.value));$("chart").addEventListener("change",e=>loadChart(e.target.value));
$("position").addEventListener("input",e=>{$("follow").checked=false;setWindow(Number(e.target.value));});
$("zoom").addEventListener("change",e=>{state.span=Number(e.target.value);if(state.song)$("position").max=Math.max(0,state.song.duration_s-state.span);setWindow(state.start);});
$("pan-left").addEventListener("click",()=>{$("follow").checked=false;setWindow(state.start-state.span/2);});$("pan-right").addEventListener("click",()=>{$("follow").checked=false;setWindow(state.start+state.span/2);});
$("seek-button").addEventListener("click",()=>{seek($("seek").value);setWindow(Number($("seek").value)-state.span/4);});$("seek").addEventListener("keydown",e=>{if(e.key==="Enter")$("seek-button").click();});
$("play").addEventListener("click",async()=>{try{if(audio.paused)await audio.play();else audio.pause();}catch(error){status(`Playback unavailable: ${error.message}`,true);}});
audio.addEventListener("play",()=>{$("play").textContent="Ⅱ Pause";$("play").setAttribute("aria-label","Pause audio");});audio.addEventListener("pause",()=>{$("play").textContent="▶ Play";$("play").setAttribute("aria-label","Play audio");});
audio.addEventListener("loadedmetadata",()=>{if(state.pendingSeek!=null){const target=state.pendingSeek;state.pendingSeek=null;seek(target);}});
audio.addEventListener("error",()=>status("Audio unavailable. Check the imported audio path and browser codec support.",true));audio.addEventListener("seeked",updateCursor);
$("overview").addEventListener("click",e=>{if(!state.song)return;const r=e.currentTarget.getBoundingClientRect(),t=(e.clientX-r.left)/r.width*state.song.duration_s;$("follow").checked=false;seek(t);setWindow(t-state.span/2);});
$("trace").addEventListener("click",e=>{const item=e.target.closest("[data-event]");if(item){selectEvent(item.dataset.event);return;}if(!state.chart)return;const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1100;if(x>=125&&x<=1080)seek(state.start+(x-125)/955*state.span);});
$("trace").addEventListener("keydown",e=>{const item=e.target.closest("[data-event]");if(item&&["Enter"," "].includes(e.key)){e.preventDefault();selectEvent(item.dataset.event);}else if(!item&&["ArrowLeft","ArrowRight"].includes(e.key)){e.preventDefault();seek(audio.currentTime+(e.key==="ArrowLeft"?-1:1));}});
$("event-table").addEventListener("click",e=>{const item=e.target.closest("[data-event]");if(item)selectEvent(item.dataset.event);});
for(const id of ["grid","estimated","measures","candidates","context"])$(id).addEventListener("change",render);
function moveNote(delta){
  const notes=state.chart?.notes.filter(n=>state.include.includes(`lane${n.column}`))||[];
  if(!notes.length)return;
  let i=notes.findIndex(n=>n.event_id===state.selected);
  i=i<0?(delta>0?0:notes.length-1):Math.max(0,Math.min(notes.length-1,i+delta));
  selectEvent(notes[i].event_id,true);
}
$("prev-note").addEventListener("click",()=>moveNote(-1));
$("next-note").addEventListener("click",()=>moveNote(1));
(async()=>{try{
  const [manifest,config]=await Promise.all([getJSON("/api/manifest"),getJSON("/api/config")]);
  state.manifest=manifest;state.include=config.include;
  if(manifest.schema_version!=="cadencia.library.v1")throw new Error("Incompatible library manifest.");
  for(const [id,layer] of [["estimated","estimated_beats"],["candidates","candidate_onsets"],["context","timing"]]){
    $(id).disabled=!state.include.includes(layer);$(id).checked=!$(id).disabled;
    $(id).parentElement.hidden=$(id).disabled;
  }
  if(!manifest.songs.length)throw new Error("No charts cached. Run uv run download_ranked_mania.py, then refresh.");
  $("corpus-size").textContent=`${manifest.songs.length} RECORDING${manifest.songs.length===1?"":"S"}`;
  $("song").innerHTML=manifest.songs.map(s=>`<option value="${esc(s.song_id)}">${esc(s.title)}</option>`).join("");$("song").disabled=false;
  const query=new URLSearchParams(location.search),first=manifest.songs.find(s=>s.song_id===query.get("song"))||manifest.songs[0];
  $("song").value=first.song_id;await loadSong(first.song_id);
}catch(error){status(error.message,true);$("coverage").textContent="No data loaded. See the library usage above.";$("import-help").open=true;}
requestAnimationFrame(frame);})();
