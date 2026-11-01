// Exercise the actual game logic with a tiny DOM/audio substitute (no browser needed).
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map(), listeners = {};
const drawing = new Proxy({}, {get: () => () => {}, set: () => true});
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    textContent:'', value:id==='speed'?'420':'', checked:false,
    classList:{toggle(){}}, setAttribute(){}, focus(){}, replaceChildren(){}, append(){},
    addEventListener(name, fn){listeners[`${id}:${name}`]=fn;},
    getContext(){return drawing;}, setPointerCapture(){},
    getBoundingClientRect(){return {left:0,width:640};},
    width:640, height:620,
  });
  return elements.get(id);
}
const audio=element('audio'); audio.paused=false; audio.ended=false; audio.currentTime=0;
const context = vm.createContext({
  document:{getElementById:element,body:element('body'),addEventListener(){},createElement(){return element('binding');}},
  window:{addEventListener(name,fn){listeners[name]=fn;}},
  requestAnimationFrame(){}, URL:{}, console,
});
const source=fs.readFileSync('src/cadencia/_simulator/app.js','utf8');
vm.runInContext(source.slice(0,source.indexOf("request('/api/config').then")),context);
function run(code){return vm.runInContext(code,context);}
run("notes=[{time_s:1,lane:0,kind:'hold',end_s:2},{time_s:1,lane:2,kind:'tap'}];reset();");
audio.currentTime=1;
run("pressLane('a',0);pressLane('b',2);");
assert.equal(run('results[0]'),'holding');assert.equal(run('results[1]'),'hit');
audio.currentTime=2;run("releaseLane('a');");
assert.equal(run('results[0]'),'hit');assert.equal(run('misses'),0);
// Early release misses exactly once.
run("reset();");audio.currentTime=1;run("pressLane('a',0);");
audio.currentTime=1.1;run("releaseLane('a');releaseLane('a');");
assert.equal(run('results[0]'),'miss');assert.equal(run('misses'),1);
// Autoplay completes both head and tail.
run('reset();');element('autoplay').checked=true;audio.currentTime=1;run('frame();');
assert.equal(run('results[0]'),'holding');audio.currentTime=2;run('frame();');
assert.equal(run('results[0]'),'hit');assert.equal(run('results[1]'),'hit');
console.log('Verified chord input, hold head/tail, early-release miss and autoplay.');
