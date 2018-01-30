import {vec4, mat4} from 'gl-matrix';
import Drawable from './Drawable';
import {gl} from '../../globals';

var activeProgram: WebGLProgram = null;

export class Shader {
  shader: WebGLShader;

  constructor(type: number, source: string) {
    this.shader = gl.createShader(type);
    gl.shaderSource(this.shader, source);
    gl.compileShader(this.shader);

    if (!gl.getShaderParameter(this.shader, gl.COMPILE_STATUS)) {
      throw gl.getShaderInfoLog(this.shader);
    }
  }
};

class ShaderProgram {
  prog: WebGLProgram;

  attrPos: number;
  attrUV: number;

  unifView: WebGLUniformLocation;
  unifViewProj: WebGLUniformLocation;
  unifinvViewProj : WebGLUniformLocation;
  unifCameraPos : WebGLUniformLocation;
  unifTimeScreen : WebGLUniformLocation;

  unifFactors : WebGLUniformLocation;
  unifFactors01 : WebGLUniformLocation;
  
  unifEnvMap00: WebGLUniformLocation;

  constructor(shaders: Array<Shader>) {
    this.prog = gl.createProgram();

    for (let shader of shaders) {
      gl.attachShader(this.prog, shader.shader);
    }
    gl.linkProgram(this.prog);
    if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) {
      throw gl.getProgramInfoLog(this.prog);
    }

    // Raymarcher only draws a quad in screen space! No other attributes
    this.attrPos = gl.getAttribLocation(this.prog, "vs_Pos");
    this.attrUV = gl.getAttribLocation(this.prog, "vs_UV");

    // TODO: add other attributes here
    this.unifView   = gl.getUniformLocation(this.prog, "u_View");
    this.unifViewProj   = gl.getUniformLocation(this.prog, "u_ViewProj");
    this.unifinvViewProj   = gl.getUniformLocation(this.prog, "u_InvViewProj");
    

    this.unifCameraPos   = gl.getUniformLocation(this.prog, "u_CameraPos");
    this.unifTimeScreen   = gl.getUniformLocation(this.prog, "u_TimeScreen");
    this.unifFactors   = gl.getUniformLocation(this.prog, "u_Factors");
    this.unifFactors01 = gl.getUniformLocation(this.prog, "u_Factors01");

    this.unifEnvMap00   = gl.getUniformLocation(this.prog, "u_EnvMap");

  }

  use() {
    if (activeProgram !== this.prog) {
      gl.useProgram(this.prog);
      activeProgram = this.prog;
    }
  }

  // TODO: add functions to modify uniforms
  setViewMatrix(vp: mat4) {
    this.use();
    if (this.unifView !== -1) {
      gl.uniformMatrix4fv(this.unifView, false, vp);
    }
  }
 
  setViewProjMatrix(vp: mat4) {
    this.use();
    if (this.unifViewProj !== -1) {
      gl.uniformMatrix4fv(this.unifViewProj, false, vp);
    }
  }

  setinvViewProjMatrix(vp: mat4) {
    this.use();
    if (this.unifinvViewProj !== -1) {
      gl.uniformMatrix4fv(this.unifinvViewProj, false, vp);
    }
  }

  setCameraPos(vec: vec4) {
    this.use();
    if (this.unifCameraPos !== -1) {
      gl.uniform4fv(this.unifCameraPos, vec);
    }
  }

  setTimeScreen(vec: vec4) {
    this.use();
    if (this.unifTimeScreen !== -1) {
      gl.uniform4fv(this.unifTimeScreen, vec);
    }
  }

  setFactors(vec: vec4) {
    this.use();
    if (this.unifFactors !== -1) {
      gl.uniform4fv(this.unifFactors, vec);
    }
  }

  setFactors01(vec: vec4) {
    this.use();
    if (this.unifFactors01 !== -1) {
      gl.uniform4fv(this.unifFactors01, vec);
    }
  }

  setEnvMap00(texture: WebGLTexture) {
    this.use();
    if (this.unifEnvMap00 !== -1) {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(gl.getUniformLocation(this.prog, "u_EnvMap"), 0);
    }
}



  draw(d: Drawable) {
    this.use();

    if (this.attrPos != -1 && d.bindPos()) {
      gl.enableVertexAttribArray(this.attrPos);
      gl.vertexAttribPointer(this.attrPos, 4, gl.FLOAT, false, 0, 0);
    }

    if (this.attrUV != -1 && d.bindUV()) {
      gl.enableVertexAttribArray(this.attrUV);
      gl.vertexAttribPointer(this.attrUV, 2, gl.FLOAT, false, 0, 0);
    }



    d.bindIdx();
    gl.drawElements(d.drawMode(), d.elemCount(), gl.UNSIGNED_INT, 0);

    if (this.attrPos != -1) gl.disableVertexAttribArray(this.attrPos);
    if (this.attrUV != -1) gl.disableVertexAttribArray(this.attrUV);

  }
};

export default ShaderProgram;
