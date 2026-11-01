'use strict';
const $=id=>document.getElementById(id), audio=$('audio'), canvas=$('game'), ctx=canvas.getContext('2d');
let config, notes=[], results=[], keys=[], held=new Set(), rebinding=null, score=0, combo=0, misses=0, duration=0, audioURL=null, loading=false;
const inputSources=new Map();
const windowS=.15, line=540, laneWidth=140, left=40;
function say(text){$('status').textContent=text;}
function label(code){return code.replace(/^Key/,'').replace(/^Digit/,'');}
function bindings(){const container=$('bindings');container.replaceChildren();keys.forEach((key,i)=>{const b=document.createElement('button');b.textContent=rebinding===i?'Press…':label(key);b.title=`Lane ${i+1}`;b.onclick=()=>{rebinding=i;bindings();};container.append(b);});}
function stats(){ $('score').textContent=score; $('combo').textContent=combo; $('misses').textContent=misses; }
function reset(){results=notes.map(()=>null);score=combo=misses=0;held.clear();inputSources.clear();stats();$('judgment').textContent='Ready when you are.';}
function award(delta){
  combo++;
  const points=Math.abs(delta)<.05?300:Math.abs(delta)<.1?100:50;
  score+=points;stats();
  $('judgment').textContent=$('autoplay').checked?'AUTOPLAY':`${points===300?'Perfect':points===100?'Great':'Good'} · ${Math.round(delta*1000)} ms`;
}
function judge(i,delta){
  if(results[i])return;
  results[i]=notes[i].kind==='hold'?'holding':'hit';
  award(delta);
}
function miss(i){results[i]='miss';misses++;combo=0;stats();$('judgment').textContent='Miss';}
function finishHold(i,delta){if(results[i]!=='holding')return;results[i]='hit';award(delta);}
function hit(lane){
  if(audio.paused||audio.ended)return;
  const t=audio.currentTime;let index=-1,best=windowS;
  notes.forEach((n,i)=>{const distance=Math.abs(n.time_s-t);if(!results[i]&&n.lane===lane&&distance<=best){best=distance;index=i;}});
  if(index>=0)judge(index,t-notes[index].time_s);
}
function pressLane(source,lane){
  if(lane<0||lane>3)return;
  inputSources.set(source,lane);held.add(lane);
  if(!$('autoplay').checked)hit(lane);
}
function releaseLane(source){
  const lane=inputSources.get(source);inputSources.delete(source);
  if(lane===undefined||[...inputSources.values()].includes(lane))return;
  held.delete(lane);
  if($('autoplay').checked)return;
  notes.forEach((n,i)=>{if(n.lane===lane&&results[i]==='holding'){
    const delta=audio.currentTime-n.end_s;
    if(Math.abs(delta)<=windowS)finishHold(i,delta);else miss(i);
  }});
}
async function play(){try{if(audio.ended){audio.currentTime=0;reset();}if(audio.paused)await audio.play();else audio.pause();}catch(e){say(`Playback failed: ${e.message}`);}}
// Always provide a viewport-filling mode, even when native fullscreen is unavailable.
const playfield=$('playfield'), fullscreenButton=$('fullscreen');
let immersive=false;
function fullscreenState(active){
  immersive=active;
  playfield.classList.toggle('immersive',active);
  document.body.classList.toggle('playing-fullscreen',active);
  fullscreenButton.textContent=active?'Exit fullscreen':'Fullscreen';
  fullscreenButton.setAttribute('aria-pressed',String(active));
}
async function exitFullscreen(){
  try{if(document.fullscreenElement===playfield)await document.exitFullscreen();}
  catch(error){$('judgment').textContent='Use your browser’s fullscreen exit control.';return;}
  fullscreenState(false);
  fullscreenButton.focus({preventScroll:true});
}
fullscreenButton.onclick=async()=>{
  if(immersive){await exitFullscreen();return;}
  fullscreenState(true);
  if(playfield.requestFullscreen){
    try{await playfield.requestFullscreen();}
    catch(error){$('judgment').textContent='Expanded play mode · browser fullscreen unavailable.';}
  }
};
document.addEventListener('fullscreenchange',()=>fullscreenState(document.fullscreenElement===playfield));
window.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&immersive&&!document.fullscreenElement){event.preventDefault();exitFullscreen();}
});
$('play').onclick=play;
$('restart').onclick=()=>{audio.pause();audio.currentTime=0;reset();};
audio.onplay=()=>{$('play').textContent='Pause';};audio.onpause=()=>{$('play').textContent='Play';held.clear();inputSources.clear();};
audio.onended=()=>{notes.forEach((note,i)=>{if(!results[i]||results[i]==='holding')miss(i);});$('play').textContent='Play again';$('judgment').textContent=`Finished · ${score} points · ${misses} misses`;};
$('song').onchange=()=>{const file=$('song').files[0];$('filename').textContent=file?file.name:'No song selected';$('generate').disabled=!file||loading;};
$('speed').oninput=()=>{$('speed-value').textContent=`${$('speed').value} px/s`;};
$('autoplay').onchange=()=>{audio.pause();audio.currentTime=0;reset();};
window.addEventListener('keydown',event=>{if(event.target.matches('input,textarea,select'))return;if(rebinding!==null){event.preventDefault();if(keys.includes(event.code)&&keys[rebinding]!==event.code){say('Each lane needs a different key.');return;}keys[rebinding]=event.code;rebinding=null;bindings();return;}const lane=keys.indexOf(event.code);if(lane<0)return;event.preventDefault();if(event.repeat)return;pressLane(event.code,lane);});
window.addEventListener('keyup',event=>releaseLane(event.code));
window.addEventListener('blur',()=>{held.clear();inputSources.clear();audio.pause();});
canvas.addEventListener('pointerdown',event=>{
  event.preventDefault();
  const box=canvas.getBoundingClientRect(),x=(event.clientX-box.left)*canvas.width/box.width;
  canvas.setPointerCapture(event.pointerId);
  pressLane(`pointer-${event.pointerId}`,Math.floor((x-left)/laneWidth));
});
for(const name of ['pointerup','pointercancel','lostpointercapture']){
  canvas.addEventListener(name,event=>releaseLane(`pointer-${event.pointerId}`));
}
async function request(url,options){const response=await fetch(url,options);const result=await response.json();if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);return result;}
async function loadPrepared(result){say(`Loading ${result.title||'saved chart'} and audio…`);const response=await fetch('/audio.wav');if(!response.ok)throw Error('Could not load prepared audio');const blob=await response.blob();if(audioURL)URL.revokeObjectURL(audioURL);audioURL=URL.createObjectURL(blob);audio.src=audioURL;notes=result.notes;duration=result.duration_s;reset();$('play').disabled=$('restart').disabled=false;const lanes=new Set(notes.map(n=>n.lane));say(`${notes.length} notes · ${lanes.size} active lanes · ${notes.filter(n=>n.kind==='hold').length} holds.${notes.length?'':' No notes were predicted.'}`);if(result.title)$('filename').textContent=result.title;}
$('generate').onclick=async()=>{const file=$('song').files[0];if(!file||loading)return;loading=true;$('generate').disabled=true;$('play').disabled=$('restart').disabled=true;audio.pause();notes=[];reset();say('Uploading and analyzing audio…');try{await request('/api/song',{method:'POST',headers:{'X-Cadencia-Token':config.token,'Content-Type':'audio/mpeg','X-Cadencia-Difficulty':$('difficulty').value,...($('bpm').value?{'X-Cadencia-Bpm':$('bpm').value}:{})},body:file});let result;do{await new Promise(resolve=>setTimeout(resolve,700));result=await request('/api/status');say('Analyzing audio and generating transformer notes…');}while(result.status==='processing');if(result.status!=='ready')throw Error(result.error||'Generation failed');await loadPrepared(result);}catch(e){say(e.message);}finally{loading=false;$('generate').disabled=false;}};
function format(t){return `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;}
function frame(){
  const t=audio.currentTime||0,speed=Number($('speed').value)||420;
  ctx.clearRect(0,0,640,620);
  for(let lane=0;lane<4;lane++){
    const x=left+lane*laneWidth;
    ctx.fillStyle=held.has(lane)?'#22483f':'#111c2b';ctx.fillRect(x+2,0,laneWidth-4,620);
    ctx.fillStyle='#8ea2ba';ctx.font='18px system-ui';ctx.textAlign='center';
    ctx.fillText(label(keys[lane]||''),x+laneWidth/2,590);
  }
  ctx.strokeStyle='#78edd1';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(left,line);ctx.lineTo(left+4*laneWidth,line);ctx.stroke();
  notes.forEach((note,i)=>{
    if(!audio.paused&&$('autoplay').checked){
      if(!results[i]&&t>=note.time_s)judge(i,0);
      if(results[i]==='holding'&&t>=note.end_s)finishHold(i,0);
    }
    if(!results[i]&&t>note.time_s+windowS)miss(i);
    if(results[i]==='holding'&&t>note.end_s+windowS)miss(i);
    if(results[i]&&results[i]!=='holding')return;
    const y=results[i]==='holding'?line:line-(note.time_s-t)*speed;
    const x=left+note.lane*laneWidth+10;
    if(note.kind==='hold'){
      const tail=line-(note.end_s-t)*speed;
      if(y>=0&&tail<=620){ctx.fillStyle='#31685e';ctx.fillRect(x+15,Math.max(0,tail),laneWidth-50,Math.min(620,y)-Math.max(0,tail));}
    }
    if(y< -20||y>620)return;
    ctx.fillStyle='#78edd1';ctx.fillRect(x,y-9,laneWidth-20,18);
  });
  $('clock').textContent=`${format(t)} / ${format(duration)}`;requestAnimationFrame(frame);
}
request('/api/config').then(async value=>{config=value;keys=[...value.keybinds];$('speed').max=Math.max(1000,value.tile_speed);$('speed').min=Math.min(100,value.tile_speed);$('speed').step='any';$('speed').value=value.tile_speed;$('speed').oninput();bindings();frame();const saved=await request('/api/status');if(saved.status==='ready')await loadPrepared(saved);}).catch(e=>say(`Could not start simulator: ${e.message}`));
