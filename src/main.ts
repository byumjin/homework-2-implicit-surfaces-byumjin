import {vec3, vec4, mat4} from 'gl-matrix';
import * as Stats from 'stats-js';
import * as DAT from 'dat-gui';
import Square from './geometry/Square';
import Triangular from './geometry/Triangular';
import Camera from './Camera';
import {setGL} from './globals';
import ShaderProgram, {Shader} from './rendering/gl/ShaderProgram';

// Define an object with application parameters and button callbacks
// This will be referred to by dat.GUI's functions that add GUI elements.
const controls = {
  // TODO: add any controls you want
  RayMarchStep : 128,
  ShadowStep : 64,
  AO : 2.0,
  SoftShadow : 8.0,
  Reflection : 1,
  MaxStep : 40,
  Intensity : 0.6,
};

let screenQuad: Square;
let triangularScreen: Triangular;

let oldTime : number;
let currentTime : number;
let elapsedTime : number;
let deltaTime : number;

function main() {

  elapsedTime = 0.0;
  oldTime = Date.now();

  // Initial display for framerate
  const stats = Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0px';
  stats.domElement.style.top = '0px';
  document.body.appendChild(stats.domElement);

  // TODO: add any controls you need to the gui
  const gui = new DAT.GUI();
  // E.G. gui.add(controls, 'tesselations', 0, 8).step(1);
  gui.add(controls, 'RayMarchStep', 32.0, 256.0).step(1.0);
  gui.add(controls, 'ShadowStep', 32.0, 256.0).step(1.0);

  gui.add(controls, 'AO', 0.0, 6.0).step(0.01);
  gui.add(controls, 'SoftShadow', 2.0, 32.0).step(1.0);

  var reflc = gui.addFolder('Reflection'); 

  reflc.add(controls, 'Reflection', { Off: 0, On: 1 });
  reflc.add(controls, 'MaxStep', 8.0, 64.0).step(1.0);
  reflc.add(controls, 'Intensity', 0.0, 10.0).step(0.01);

  

  // get canvas and webgl context
  const canvas = <HTMLCanvasElement> document.getElementById('canvas');

  function setSize(width: number, height: number) {
    canvas.width = width;
    canvas.height = height;
  }

  const gl = <WebGL2RenderingContext> canvas.getContext('webgl2');
  if (!gl) {
    alert('WebGL 2 not supported!');
  }
  // `setGL` is a function imported above which sets the value of `gl` in the `globals.ts` module.
  // Later, we can import `gl` from `globals.ts` to access it
  setGL(gl);

  triangularScreen = new Triangular(vec3.fromValues(0, 0, 0));
  triangularScreen.create();
  triangularScreen.bindEnvMap00("src/textures/canivalBG.png");

  const camera = new Camera(vec3.fromValues(0, 0.0, 9.0), vec3.fromValues(0, 0, 0));

  gl.clearColor(0.0, 0.0, 0.0, 1);
  gl.disable(gl.DEPTH_TEST);

  const raymarchShader = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/screenspace-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/raymarch-frag.glsl')),
  ]);

 

  function updateTime()
  {
    currentTime = Date.now();

    deltaTime = currentTime - oldTime;
    elapsedTime += deltaTime;    

    oldTime = currentTime;
  }

  // This function will be called every frame
  function tick() {

    updateTime();

    camera.update();
    stats.begin();

    gl.viewport(0, 0, window.innerWidth, window.innerHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // TODO: get / calculate relevant uniforms to send to shader here
    // TODO: send uniforms to shader

    let viewProj = mat4.create();
    mat4.multiply(viewProj, camera.projectionMatrix, camera.viewMatrix);

    let invViewProj = mat4.create();
    mat4.invert(invViewProj, viewProj);

    raymarchShader.setViewMatrix(camera.viewMatrix);
    raymarchShader.setViewProjMatrix(viewProj);
    raymarchShader.setinvViewProjMatrix(invViewProj);
    raymarchShader.setCameraPos(vec4.fromValues( camera.position[0], camera.position[1], camera.position[2], 1.0));
    raymarchShader.setTimeScreen(vec4.fromValues( elapsedTime, elapsedTime, window.innerWidth, window.innerHeight));
    raymarchShader.setFactors(vec4.fromValues( controls.AO, controls.SoftShadow, controls.RayMarchStep, controls.ShadowStep));
    raymarchShader.setFactors01(vec4.fromValues( controls.MaxStep, controls.Reflection, controls.Intensity, controls.ShadowStep));

    raymarchShader.setEnvMap00(triangularScreen.envMap00);

    // March!
    raymarchShader.draw(triangularScreen);

    // TODO: more shaders to layer / process the first one? (either via framebuffers or blending)

    stats.end();

    // Tell the browser to call `tick` again whenever it renders a new frame
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', function() {
    setSize(window.innerWidth, window.innerHeight);
    camera.setAspectRatio(window.innerWidth / window.innerHeight);
    camera.updateProjectionMatrix();
  }, false);

  setSize(window.innerWidth, window.innerHeight);
  camera.setAspectRatio(window.innerWidth / window.innerHeight);
  camera.updateProjectionMatrix();

  // Start the render loop
  tick();
}

main();
