#version 300 es

precision highp float;

#define MAX_MARCHING_STEPS 128
#define MAX_SHADOW_STEPS 128
#define MAX_REFLECTION_STEPS 30
#define MAX_RAYDISTANCE 1000.0
#define SHADOW_EPSILON 0.0001
#define USE_CHEAP_NORMAL 1

#define STEP_SIZE_SCALER 0.8

#define PI 3.1415926535897932384626422832795028841971
#define TwoPi 6.28318530717958647692
#define InvPi 0.31830988618379067154
#define Inv2Pi 0.15915494309189533577
#define Inv4Pi 0.07957747154594766788

#define DEGREE_TO_RAD 0.01745329251994329576923690768489

#define DEGREE_10 0.17453292519943295769236907684886
#define DEGREE_12 0.20943951023931954923084289221863
#define DEGREE_18_5 0.32288591161895097173088279217039
#define DEGREE_24 0.41887902047863909846168578443727
#define DEGREE_30 0.52359877559829887307710723054658
#define DEGREE_40 0.69813170079773183076947630739545
#define DEGREE_60 1.0471975511965977461542144610932
#define DEGREE_80 1.3962634015954636615389526147909
#define DEGREE_90 1.5707963267948966192313216916398
#define DEGREE_72 1.2566370614359172953850573533118
#define DEGREE_100 1.7453292519943295769236907684886
#define DEGREE_140 2.4434609527920614076931670758841
#define DEGREE_144 2.5132741228718345907701147066236
#define DEGREE_160 2.7925268031909273230779052295818

#define PLANE 1.0
#define FLOOR00 2.0
#define FLOOR001 2.1
#define FLOOR01 4.0
#define BALLOON00 6.0
#define SLOPES 8.0

#define BODY 10.0
#define FOOT 12.0
#define MOUTH 14.0
#define TONGUE 16.0
#define OUTER_EYE 18.0
#define INNER_EYE 20.0
#define CHIN 22.0

#define STAR_PANNEL 24.0
#define STAR_BG 26.0

uniform mat4 u_View;
uniform mat4 u_ViewProj;
uniform mat4 u_InvViewProj;

uniform vec4 u_CameraPos;

uniform vec4 u_TimeScreen;
uniform vec4 u_Factors; //x : AO, y : SoftShadow
uniform vec4 u_Factors01;

uniform sampler2D u_EnvMap;

in vec2 fs_UV;

out vec4 out_Col;


vec2 LightingFunGGX_FV(float dotLH, float roughness)
{
	float alpha = roughness*roughness;

	//F
	float F_a, F_b;
	float dotLH5 = pow(clamp(1.0f - dotLH, 0.0f, 1.0f), 5.0f);
	F_a = 1.0f;
	F_b = dotLH5;

	//V
	float vis;
	float k = alpha * 0.5f;
	float k2 = k*k;
	float invK2 = 1.0f - k2;
	vis = 1.0f/(dotLH*dotLH*invK2 + k2);

	return vec2((F_a - F_b)*vis, F_b*vis);
}

float LightingFuncGGX_D(float dotNH, float roughness)
{
	float alpha = roughness*roughness;
	float alphaSqr = alpha*alpha;
	float denom = dotNH * dotNH * (alphaSqr - 1.0f) + 1.0f;

	return alphaSqr / (PI*denom*denom);
}

vec3 GGX_Spec(vec3 Normal, vec3 HalfVec, float Roughness, vec3 BaseColor, vec3 SpecularColor, vec2 paraFV)
{
	float NoH = clamp(dot(Normal, HalfVec), 0.0, 1.0);

	float D = LightingFuncGGX_D(NoH * NoH * NoH * NoH, Roughness);
	vec2 FV_helper = paraFV;

	vec3 F0 = SpecularColor;
	vec3 FV = F0*FV_helper.x + vec3(FV_helper.y, FV_helper.y, FV_helper.y);
	
	return D * FV;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }
float noise(float x) { float i = floor(x); float f = fract(x); float u = f * f * (3.0 - 2.0 * f); return mix(hash(i), hash(i + 1.0), u); }
float noise(vec2 x) { vec2 i = floor(x); vec2 f = fract(x); float a = hash(i); float b = hash(i + vec2(1.0, 0.0)); float c = hash(i + vec2(0.0, 1.0)); float d = hash(i + vec2(1.0, 1.0)); vec2 u = f * f * (3.0 - 2.0 * f); return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y; }

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

mat4 scaleMatrix(vec3 scale)
{    
    return mat4(scale.x, 0.0,  0.0,  0.0,
                0.0,  scale.y, 0.0,  0.0,
                0.0,  0.0,  scale.z, 0.0,
                0.0,  0.0,      0.0, 1.0);
}

// polynomial smooth min (k = 0.1);
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

vec2 opU( vec2 d1, vec2 d2 )
{
	return d1.x >= d2.x ?  d2 : d1;
}

vec2 opsU( vec2 d1, vec2 d2, float k )
{
    return vec2(smin(d1.x, d2.x, k), d1.x > d2.x ?  d2.y : d1.y);
}

vec2 opS( vec2 d1, vec2 d2, bool d1color )
{
   return vec2( max(-d1.x,d2.x), d1color ? d1.y : d2.y );
}

vec3 opScale( vec3 p, vec3 s )
{
    return vec3(p.x/s.x, p.y/s.y, p.z/s.z);
}

vec3 opTwist( vec3 p )
{
	float twisterFactor = 1.0;

    float  c = cos(twisterFactor*p.y+twisterFactor);
    float  s = sin(twisterFactor*p.y+twisterFactor);
    mat2   m = mat2(c,-s,s,c);
    return vec3(m*p.xz,p.y);
}

float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}

float sdDisk( vec3 p, vec4 n, float radius)
{
  float gap = p.x * p.x + p.z * p.z - radius;

  // n must be normalized
  return dot(p,n.xyz) + n.w - gap;
}

float sdTorus( vec3 p, vec2 t )
{
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

// for Test
float mad(vec3 p)
{
    vec3 w = p;
    float m = dot(w,w);

    vec4 trap = vec4(abs(w),m);
	float dz = 1.0;
    
	for( int i=0; i<4; i++ )
    {
        float m2 = m*m;
        float m4 = m2*m2;
		dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

        float x = w.x; float x2 = x*x; float x4 = x2*x2;
        float y = w.y; float y2 = y*y; float y4 = y2*y2;
        float z = w.z; float z2 = z*z; float z4 = z2*z2;

        float k3 = x2 + z2;
        float k2 = inversesqrt( k3*k3*k3*k3*k3*k3*k3 );
        float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
        float k4 = x2 - y2 + z2;

        w.x = p.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
        w.y = p.y + -16.0*y2*k3*k4*k4 + k1*k1;
        w.z = p.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;
        
        trap = min( trap, vec4(abs(w),m) );

        m = dot(w,w);
		if( m > 256.0 )
            break;
    }

    return 0.25*log(m)*sqrt(m)/dz;
}

float sdCappedCone( in vec3 p, in vec3 c )
{
    vec2 q = vec2( length(p.xz), p.y );
    vec2 v = vec2( c.z*c.y/c.x, -c.z );
    vec2 w = v - q;
    vec2 vv = vec2( dot(v,v), v.x*v.x );
    vec2 qv = vec2( dot(v,w), v.x*w.x );
    vec2 d = max(qv,0.0)*qv/vv;
    return sqrt( dot(w,w) - max(d.x,d.y) ) * sign(max(q.y*v.x-q.x*v.y,w.y));
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

float sdEllipsoid( in vec3 p, in vec3 r )
{
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}

float sdCone( in vec3 p, in vec3 c )
{
    vec2 q = vec2( length(p.xz), p.y );
    float d1 = -q.y-c.z;
    float d2 = max( dot(q,c.xy), q.y);
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
	vec3 pa = p-a, ba = b-a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - r;
}

float sdCylinder( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float udRoundBox( vec3 p, vec3 b, float r )
{
    return length(max(abs(p)-b,0.0))-r;
}
vec3 Blend_Thorn(vec3 q, float k)
{
	//Bend
	float c = cos(k*q.y);
    float s = sin(k*q.y);
    mat2  m = mat2(c,-s,s,c);
    return vec3(q.x, m*q.yz);    
}

vec3 Blend_ThornZ(vec3 q, float k)
{
	//Bend
	float c = cos(k*q.y);
    float s = sin(k*q.y);
    mat2  m = mat2(c,-s,s,c);
    return vec3(m*q.xy, q.z);    
}

vec2 balloon00( vec3 p )
{
	//if(p.y > 40.0 || p.y < -5.0)
	//	return vec2(1000.0, -1.0);
	
	vec3 pos = p;
	

	vec3 c = vec3(124.0, 150.0, 89.0);
	vec3 q = mod(pos,c)-0.5*c;
    return  vec2(sdSphere( q, 4.0 ), BALLOON00);
}

vec2 floor00( vec3 p)
{
	//if(p.y > 0.0)
	//	return vec2(1000.0, -1.0);

	vec3 pos = p - vec3(0.0, 175.0, 0.0);

	vec3 c = vec3(80.0, 400.0, 200.0);


	pos.z += int( pos.x / c.x) % 2 == 0 ? u_TimeScreen.x * 0.01 : -u_TimeScreen.x * 0.01;

	
	vec3 q = mod(pos,c)-0.5*c;
    return  vec2(sdBox(q, vec3(10.0, 4.0, 24.0)), FLOOR00);
	//opS(vec2(sdBox(q - vec3(0.0, 10.0, 0.0), vec3(5.0, 0.8, 28.0)), FLOOR001),vec2(sdBox(q, vec3(10.0, 4.0, 24.0)), FLOOR00));
}

vec2 floor01( vec3 p)
{
	if(p.y > 0.0)
		return vec2(1000.0, -1.0);

	vec3 pos = p;

	vec3 c = vec3(250.0, 100.0, 100.0);
	vec3 q = mod(pos,c)-0.5*c;
    return vec2(sdBox(q, vec3(60.0, 2.0, 25.0)), FLOOR01);
}

vec2 slopes(vec3 p)
{
	vec3 pos = p;

	mat4 rot = rotationMatrix(vec3(1.0, 0.0, 1.0), DEGREE_TO_RAD * 45.0 );
	vec3 q = inverse(mat3(rot))* (pos);	

	vec3 c = vec3(350.0, 20.0, 240.0);

	q = mod(q,c)-0.5*c;

	

	return  vec2(sdCylinder(q, vec2(5.0,100.0) ), SLOPES);
}

float Spring(vec3 p)
{
	float u = 0.0;
	float v = 0.0;

	float r1 = 0.5;
	float r2 = 0.5;
	float periodlength = 1.5;

	float x = (1.0 - r1 * cos(v)) * cos(u);
	float y = (1.0 - r1 * cos(v)) * sin(u);
	float z = r2 * (sin(v) + periodlength * u / PI);

	return p.x - x + p.y - y + p.z - z;
}

vec2 Twister(vec3 p)
{
	vec3 pos = p;

	pos.z += 10.0;

	return vec2(Spring(pos), 90.0);
}

vec2 boundingSphere( in vec4 sph, in vec3 ro, in vec3 rd )
{
    vec3 oc = ro - sph.xyz;
    
	float b = dot(oc,rd);
	float c = dot(oc,oc) - sph.w*sph.w;
    float h = b*b - c;
    
    if( h<0.0 ) return vec2(-1.0);

    h = sqrt( h );

    return -b + vec2(-h,h);
}


vec2 kirby( vec3 p, vec3 ro, vec3 rd, float upDown)
{
  // bounding sphere
  
  vec2 dis = boundingSphere( vec4(0.0, -upDown,0.0,1.6), ro, rd );

  if(dis.y < 0.0)
	return vec2(1000.0, -1.0);  

  //Body
  vec2 body = vec2( sdSphere(p, 1.0), BODY );

  //Mouth
  mat4 rot_m00 = rotationMatrix(vec3(0.0, 1.0, 1.0), DEGREE_90 );
  mat4 rot_m01 = rotationMatrix(vec3(1.0, 0.0, 0.0), -DEGREE_90 ) * rot_m00;
  vec3 m_00 = inverse(mat3(rot_m00))* (p - vec3( 0.0, -0.25, 1.1));
  vec2 mouth = vec2( udRoundBox(m_00, vec3(0.08), 0.07), MOUTH );

  //Tongue
  vec2 tongue = vec2( sdSphere(p - vec3( 0.0, -0.25, 0.75), 0.2), TONGUE );

  //LeftEye
  mat4 rot_le00 = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_12 );
  mat4 rot_le01 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_12 ) * rot_le00;
  vec3 le_00 = inverse(mat3(rot_le01))* (p - vec3( 0.19, 0.2, 0.949));
  le_00 = Blend_Thorn(le_00, -0.4);
  vec2 leftEye = vec2( sdEllipsoid(le_00 , vec3(0.08, 0.25, 0.03)) , OUTER_EYE );

   //RightEye
  //mat4 rot_re00 = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_12 );
  mat4 rot_re01 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_12 ) * rot_le00;
  vec3 re_00 = inverse(mat3(rot_re01))* (p - vec3( -0.19, 0.2, 0.949));
  re_00 = Blend_Thorn(re_00, -0.4);
  vec2 rightEye = vec2( sdEllipsoid(re_00 , vec3(0.08, 0.25, 0.03)) , OUTER_EYE );

  //LeftInnerEye
  mat4 rot_lie00 = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_18_5 );
  mat4 rot_lie01 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_12 ) * rot_lie00;
  vec3 lie_00 = inverse(mat3(rot_lie01))* (p - vec3( 0.19, 0.3, 0.949));
  lie_00 = Blend_Thorn(lie_00, -0.4);
  vec2 leftInnerEye = vec2( sdEllipsoid(lie_00 , vec3(0.05, 0.09, 0.01)) , INNER_EYE );

  //RightInnerEye
  //mat4 rot_rie00 = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_18_5 );
  mat4 rot_rie01 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_12 ) * rot_lie00;
  vec3 rie_00 = inverse(mat3(rot_rie01))* (p - vec3( -0.19, 0.3, 0.949));
  rie_00 = Blend_Thorn(rie_00, -0.4);
  vec2 rightInnerEye = vec2( sdEllipsoid(rie_00 , vec3(0.05, 0.09, 0.01)) , INNER_EYE );

  // Pinky Chin

  mat4 rot_rc00 = rotationMatrix(vec3(0.0, 0.0, 1.0), DEGREE_90 );
  mat4 rot_rc01 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_24 ) * rot_rc00;
  mat4 rot_rc02 = rotationMatrix(vec3(1.0, 0.0, 0.0), -DEGREE_10 ) * rot_rc01;
  vec3 rc_00 = inverse(mat3(rot_rc02))* (p - vec3( -0.4, -0.15, 0.905));
  rc_00 = Blend_Thorn(rc_00, -0.4);
  vec2 rightChin = vec2( sdEllipsoid(rc_00 , vec3(0.05, 0.09, 0.01)) , CHIN );
  
  //mat4 rot_lc00 = rotationMatrix(vec3(0.0, 0.0, 1.0), DEGREE_TO_RAD * 90.0 );
  mat4 rot_lc01 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_24 ) * rot_rc00;
  mat4 rot_lc02 = rotationMatrix(vec3(1.0, 0.0, 0.0), -DEGREE_10 ) * rot_lc01;
  vec3 lc_00 = inverse(mat3(rot_lc02))* (p - vec3( 0.4, -0.15, 0.905));
  lc_00 = Blend_Thorn(lc_00, -0.4);
  vec2 leftChin = vec2( sdEllipsoid(lc_00 , vec3(0.05, 0.09, 0.01)) , CHIN );


  float rightArmSeed = cos(u_TimeScreen.x * 0.03)*0.4;

  //Right Arm
  mat4 rot_00 = rotationMatrix(vec3(0.0, 0.0, 1.0), DEGREE_140 + rightArmSeed);
  vec3 q_00 = inverse(mat3(rot_00))* (p - vec3(-0.8, 0.3, 0.0));
  vec2 rightArm = vec2( sdEllipsoid(q_00, vec3(0.4, 0.6, 0.4)) , BODY );

  //Left Arm
  mat4 rot_01 = rotationMatrix(vec3(-1.0, 0.0, 0.0), -DEGREE_30 );
  vec3 q_01 = inverse(mat3(rot_01))* (p - vec3(0.65, -0.35, 0.4));
  vec2 leftArm = vec2( sdEllipsoid(q_01, vec3(0.35, 0.6, 0.35)) , BODY );
  
  float footSeed = sin(u_TimeScreen.x * 0.01)*0.2;

  //Right Foot
  mat4 rot_02 = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_160 );
  mat4 rot_021 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_80 ) * rot_02;
  mat4 rot_022 = rotationMatrix(vec3(0.0, 0.0, 1.0), -DEGREE_40 + footSeed ) * rot_021;
  vec3 q_02 = inverse(mat3(rot_022))* (p - vec3(-0.8, -0.6, 0.4));
  q_02 = Blend_Thorn(q_02, -1.0);
  vec2 rightFoot = vec2( sdEllipsoid(q_02, vec3(0.35, 0.6, 0.35)) , FOOT );

  //Left Foot
  //mat4 rot_03 = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_160 );
  //mat4 rot_031 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_80 + footSeed ) * rot_02;
  mat4 rot_032 = rotationMatrix(vec3(0.0, 0.0, 1.0), -DEGREE_100 - footSeed ) * rot_021;
  vec3 q_03 = inverse(mat3(rot_032))* (p - vec3(0.8, -0.6, 0.2));
  q_03 = Blend_Thorn(q_03, -1.0);
  vec2 leftFoot = vec2( sdEllipsoid(q_03, vec3(0.35, 0.6, 0.35)) , FOOT);


  vec2 result = opS(mouth, body, false);

  result = opU(result, leftEye);
  result = opU(result, rightEye);

  result = opU(result, leftInnerEye);
  result = opU(result, rightInnerEye);
  
  result = opU(result, rightChin);
  result = opU(result, leftChin);
  

  result = opU(result, tongue);
  result = opsU(result, rightArm, 0.02);
  result = opsU(result, leftArm, 0.02);
  result = opU(result, rightFoot);
  result = opU(result, leftFoot);

  return result;
}

vec2 star( vec3 p, vec3 ro, vec3 rd, float upDown)
{
  // bounding sphere
  vec2 dis = boundingSphere( vec4(0.0, -upDown,0.0,2.4), ro, rd );

  if(dis.y < 0.0)
	return vec2(1000.0, -1.0);

 
  vec3 pos = p - vec3(0.0, -1.7, 0.0);

  

  mat4 globalRot = rotationMatrix(vec3(0.0, 1.0, 0.0), u_TimeScreen.x * 0.001 );

  pos = mat3(globalRot) * pos;

 

  float radius = 1.4;

  vec3 starShape = vec3(0.15, 0.7, 0.3);

  float bendFactor = -2.0;

  mat4 rot_star00 = rotationMatrix(vec3(0.0, 0.0, 1.0), DEGREE_90 );
  vec3 s_00 = inverse(mat3(rot_star00))* (pos - vec3(0.0, 0.0, radius));
  s_00 = Blend_Thorn(s_00, bendFactor);

  vec2 star00 = vec2( sdEllipsoid(s_00, starShape), STAR_PANNEL );

  mat4 rot_star01 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_72 ) * rot_star00;
  vec3 s_01 = inverse(mat3(rot_star01))* (pos - vec3(-0.95105651629515357211643933337938 * radius, 0.0, 0.30901699437494742410229341718282 * radius));
  s_01 = Blend_Thorn(s_01, bendFactor);
  vec2 star01 = vec2( sdEllipsoid(s_01, starShape), STAR_PANNEL );

  mat4 rot_star02 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_72 ) * rot_star00;
  vec3 s_02 = inverse(mat3(rot_star02))* (pos - vec3(0.95105651629515357211643933337938 * radius, 0.0, 0.30901699437494742410229341718282 * radius));
  s_02 = Blend_Thorn(s_02, bendFactor);
  vec2 star02 = vec2( sdEllipsoid(s_02, starShape), STAR_PANNEL );

  mat4 rot_star03 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_144 ) * rot_star00;
  vec3 s_03 = inverse(mat3(rot_star03))* (pos - vec3(-0.58778525229247312916870595463907 * radius, 0.0, -0.80901699437494742410229341718282 * radius));
  s_03 = Blend_Thorn(s_03, bendFactor);
  vec2 star03 = vec2( sdEllipsoid(s_03, starShape), STAR_PANNEL );

  mat4 rot_star04 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_144 ) * rot_star00;
  vec3 s_04 = inverse(mat3(rot_star04))* (pos - vec3(0.58778525229247312916870595463907 * radius, 0.0, -0.80901699437494742410229341718282 * radius));
  s_04 = Blend_Thorn(s_04, bendFactor);
  vec2 star04 = vec2( sdEllipsoid(s_04, starShape), STAR_PANNEL );

  vec2 center = vec2( sdEllipsoid(pos,  vec3(0.7, 0.4, 0.7) * radius), STAR_PANNEL );
 
  float k = 0.3;

  vec2 result = opsU(star00, star01, k);
  result = opsU(result, star02, k);
  result = opsU(result, star03, k);
  result = opsU(result, star04, k);
  result = opsU(result, center, 0.5);

  return result;
}


vec2 starBG( vec3 p, vec3 ro, vec3 rd, vec3 oriPos, vec3 gapPos)
{
  // bounding sphere
  //vec2 dis = boundingSphere( vec4(gapPos, 2.4), ro, rd );

  //if(dis.y < 0.0)
  // return vec2(1000.0, -1.0);

  if(oriPos.y > 20.0 || oriPos.y < -10.0 || oriPos.z < -10.0 || oriPos.z > 10.0  || oriPos.x < -30.0 || oriPos.x > 30.0 )
   return vec2(1000.0, -1.0);

 
  vec3 pos = p;

  

  mat4 globalRot = rotationMatrix(vec3(1.0, 0.0, 0.0), DEGREE_90 );
  globalRot = rotationMatrix(vec3(0.0, 1.0, 0.0), u_TimeScreen.x * 0.001 )  * globalRot;
  pos = mat3(globalRot) * pos;

 

  float radius = 1.4;

  float transSeed = (sin(u_TimeScreen.x * 0.001) + 1.0) * 0.3;

  vec3 starShape = vec3(0.15, 0.7, 0.3 + transSeed);

  float bendFactor = -2.0;

  mat4 rot_star00 = rotationMatrix(vec3(0.0, 0.0, 1.0), DEGREE_90 );
  vec3 s_00 = inverse(mat3(rot_star00))* (pos - vec3(0.0, 0.0, radius));
  s_00 = Blend_Thorn(s_00, bendFactor);

  vec2 star00 = vec2( sdEllipsoid(s_00, starShape), STAR_BG );

  mat4 rot_star01 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_72 ) * rot_star00;
  vec3 s_01 = inverse(mat3(rot_star01))* (pos - vec3(-0.95105651629515357211643933337938 * radius, 0.0, 0.30901699437494742410229341718282 * radius));
  s_01 = Blend_Thorn(s_01, bendFactor);
  vec2 star01 = vec2( sdEllipsoid(s_01, starShape), STAR_BG );

  mat4 rot_star02 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_72 ) * rot_star00;
  vec3 s_02 = inverse(mat3(rot_star02))* (pos - vec3(0.95105651629515357211643933337938 * radius, 0.0, 0.30901699437494742410229341718282 * radius));
  s_02 = Blend_Thorn(s_02, bendFactor);
  vec2 star02 = vec2( sdEllipsoid(s_02, starShape), STAR_BG );

  mat4 rot_star03 = rotationMatrix(vec3(0.0, 1.0, 0.0), DEGREE_144 ) * rot_star00;
  vec3 s_03 = inverse(mat3(rot_star03))* (pos - vec3(-0.58778525229247312916870595463907 * radius, 0.0, -0.80901699437494742410229341718282 * radius));
  s_03 = Blend_Thorn(s_03, bendFactor);
  vec2 star03 = vec2( sdEllipsoid(s_03, starShape), STAR_BG );

  mat4 rot_star04 = rotationMatrix(vec3(0.0, 1.0, 0.0), -DEGREE_144 ) * rot_star00;
  vec3 s_04 = inverse(mat3(rot_star04))* (pos - vec3(0.58778525229247312916870595463907 * radius, 0.0, -0.80901699437494742410229341718282 * radius));
  s_04 = Blend_Thorn(s_04, bendFactor);
  vec2 star04 = vec2( sdEllipsoid(s_04, starShape), STAR_BG );

  vec2 center = vec2( sdEllipsoid(pos,  vec3(0.7, 0.4, 0.7) * radius), STAR_BG );
 
  float k = 0.3;

  vec2 result = opsU(star00, star01, k);
  result = opsU(result, star02, k);
  result = opsU(result, star03, k);
  result = opsU(result, star04, k);
  result = opsU(result, center, 0.5);

  return result;
}


vec2 stage( vec3 p, vec3 ro, vec3 rd)
{
  // bounding sphere
  vec2 dis = boundingSphere( vec4(0.0, 0.0, 0.0, 11.0), ro, rd );
  if(dis.y < 0.0)
   return vec2(1000.0, -1.0);

  return vec2( sdCylinder( p, vec2(8.0, 0.2) ), PLANE);

}

vec2 SDF( vec3 p, vec3 ro, vec3 rd )
{
  vec2 result;

  float upDown = sin(u_TimeScreen.x * 0.00173) * 0.5;

  vec3 pos = p;
  pos.y += upDown;

  result = kirby(pos, ro, rd, upDown);
  result = opU(result, star(pos, ro, rd, upDown));

  //repeat BG obj
  vec3 bg_pos = p;
  bg_pos.xy += vec2(u_TimeScreen.x * 0.0051173);
  bg_pos.x += cos( u_TimeScreen.x * 0.004173) * 4.0;

  vec3 c = vec3(20.0, 20.0, 20.0);
  bg_pos = mod(bg_pos,c)-0.5*c;
  
  vec3 gapPos = bg_pos - p;  

  result = opU(result, starBG(bg_pos, ro, rd, p, gapPos));  
  result = opU(result, stage(p - vec3(0.0, -3.0, 0.0), ro, rd));
  
  return result;
}

// http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
float checkersGradBox( in vec2 p )
{
    // filter kernel
    vec2 w = fwidth(p) + 0.001;
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;                  
}

float stripsRedWhite( in vec2 p )
{
    // filter kernel
    vec2 w = fwidth(p) + 0.001;

	// analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;

    return i.y;                  
}

vec3 getSurfaceNormal(vec3 endPoint, float epsilonParam, vec3 ro, vec3 rd)
{
	float epsilon = epsilonParam;

#if USE_CHEAP_NORMAL == 0
	// 6 samples
	return normalize( vec3( SDF(vec3(endPoint.x + epsilon, endPoint.y, endPoint.z), ro, rd).x -  SDF(vec3(endPoint.x - epsilon, endPoint.y, endPoint.z), ro, rd).x,
							SDF(vec3(endPoint.x, endPoint.y + epsilon, endPoint.z), ro, rd).x -  SDF(vec3(endPoint.x, endPoint.y - epsilon, endPoint.z), ro, rd).x,
							SDF(vec3(endPoint.x, endPoint.y, endPoint.z + epsilon), ro, rd).x -  SDF(vec3(endPoint.x, endPoint.y, endPoint.z - epsilon), ro, rd).x));
	
#else
	// 4 samples
	vec2 e = vec2(1.0,-1.0)*0.5773*epsilon;
	return normalize( e.xyy*SDF( endPoint + e.xyy, ro, rd).x + 
					  e.yyx*SDF( endPoint + e.yyx, ro, rd).x + 
					  e.yxy*SDF( endPoint + e.yxy, ro, rd).x + 
					  e.xxx*SDF( endPoint + e.xxx, ro, rd).x );	
#endif

}

float shadow( vec3 endPoint, vec3 lightVec, float mint, float k, float epsilon)
{
	float shadowFactor = 1.0;

	float depth = mint;

	int maxShadowStep = int(u_Factors.w);

	for(int i=0; i< maxShadowStep; i++)
    {
        float dist = SDF(endPoint + lightVec * depth, endPoint, lightVec).x;

        
		if( dist < epsilon * depth)
            return 0.0;
        

		shadowFactor = min( shadowFactor, u_Factors.y * dist/ depth );

		dist = min(dist, 0.02);

        depth += dist * STEP_SIZE_SCALER;
    }
    return clamp(shadowFactor, 0.0, 1.0);
}

vec4 reflection( vec3 endPoint, vec3 reflectVec, float epsilon)
{
	float depth = 0.1;
	
	
	int maxStep = int(u_Factors01.x);

	for(int i=0; i< maxStep; i++)
    {
        vec2 result = SDF(endPoint + reflectVec * depth, endPoint, reflectVec);
        float dist = result.x;
        
		if( dist < epsilon * depth)
		{
			return vec4(endPoint, result.y);
		}

        depth += dist * STEP_SIZE_SCALER;
    }
    return vec4(endPoint, -1.0);
}

//5 tap AO
float getAO(vec3 endPoint, vec3 normal)
{
	float stepLen = u_CameraPos.w;
	float AO = 0.0;
    float att = 1.0;

    float offset = 0.02;
   
    for( int i=0; i<5; i++ )
    {
        float dist = offset + stepLen*float(i)/4.0;
        vec3 newEndpoint =  normal * dist + endPoint;
        vec2 VAL = SDF( newEndpoint, endPoint, normal );

        //skip when it reachs these surfaces
        if( OUTER_EYE - 0.5 < VAL.y && CHIN + 0.5 > VAL.y)
        {
            
        }
        else
        {
            float gap = (dist - VAL.x);
            AO += gap*att;
        }

        att *= 0.95;
    }

	return 1.0 - clamp(u_Factors.x * AO, 0.0, 1.0);
}

vec4 rayMarching(vec3 viewVec, vec3 eyePos, vec3 lightVec, out bool isHit, out vec3 normal, float epsilon, out float AO, out float shadowFactor, out vec4 reflectInfo)
{
	isHit = false;
	float depth = 0.1;

	int count = 0;

	vec3 endPoint;

	float radius = 1.0;
	vec3 c = vec3(10.0);

	int maxRayStep = int(u_Factors.z);

	for(int i=0; i<maxRayStep; i++)
	{
		endPoint = eyePos + depth * viewVec;

		vec2 result = SDF( endPoint, eyePos, viewVec);

		float dist = result.x;

		if(dist < epsilon * depth) 
		{
			isHit = true;

			shadowFactor = shadow(endPoint, lightVec, epsilon, 8.0, epsilon);

			//shadowFactor = 1.0;

			if(shadowFactor < 0.0)
				return vec4(endPoint, -1.0);

			normal = getSurfaceNormal(endPoint, epsilon, eyePos, viewVec);

			if(u_Factors01.y > 0.5)
			{
				vec3 reflectVec = reflect(viewVec, normal);
				reflectInfo = reflection(endPoint, reflectVec, epsilon);
			}		

			AO = getAO(endPoint, normal);

			return vec4(endPoint, result.y);
		}

		depth += dist * STEP_SIZE_SCALER;// + epsilon * log(float(i) + 1.0);

		if(depth >= MAX_RAYDISTANCE)
		{			
			return vec4(endPoint, -1.0);
		}
	}

	return vec4(endPoint, -1.0);
}

vec3 addStars(vec2 screenSize)
{
    float time = u_TimeScreen.x * 0.08;

    // Background starfield
    float galaxyClump = (pow(noise(fs_UV.xy * (30.0 * screenSize.x)), 3.0) * 0.5 + pow(noise(100.0 + fs_UV.xy * (15.0 * screenSize.x)), 5.0)) / 3.5;
    
    vec3 starColor = vec3(galaxyClump * pow(hash(fs_UV.xy), 1500.0) * 80.0);

    starColor.x *= sqrt(noise(fs_UV.xy) * 1.2);
    starColor.y *= sqrt(noise(fs_UV.xy * 4.0));

    vec2 delta = (fs_UV.xy - screenSize.xy * 0.5) * screenSize.y * 1.2;  
    float radialNoise = mix(1.0, noise(normalize(delta) * 20.0 + time * 0.5), 0.12);

    float att = 0.057 * pow(max(0.0, 1.0 - (length(delta) - 0.9) / 0.9), 8.0);

    starColor += radialNoise * min(1.0, att);

    float randSeed = rand(fs_UV);

    return starColor *  (( sin(randSeed + randSeed * time* 0.05) + 1.0)* 0.4 + 0.2);
}

vec2 getNDC(vec2 uv)
{
	return vec2(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
}

vec3 getViewVecFromScreenSpace(vec2 uv, vec3 eyePos)
{
	vec2 NDC = getNDC(fs_UV);

	vec4 farPos = vec4(NDC, 1.0, 1.0);
	farPos = u_InvViewProj * farPos;
	farPos /= farPos.w;

	return  normalize(farPos.xyz - eyePos);
}

float SphericalTheta(vec3 v)
{
	return acos(clamp(v.y, -1.0f, 1.0f));
}

float SphericalPhi(vec3 v)
{
	float p = atan(v.z , v.x);
	return (p < 0.0f) ? (p + TwoPi) : p;
}

void getSurfaceColor(in float materialFator, vec3 endPoint, out vec4 BasicColor, out float Roughness  )
{
	if(materialFator < 0.0)
	{
		BasicColor = vec4(0.0);
		Roughness = 1.0;
	}
	else if( materialFator < PLANE + 0.5 )
	{
		float checker = checkersGradBox(vec2(endPoint.x, endPoint.z));	
        if(checker > 0.5)
        {
            BasicColor = vec4(1.0, 0.541176, 0.639216, 1.0);
        }
        else
        {
            BasicColor = vec4(1.0, 1.0, 0.3, 1.0);
        }

		//checker = max(checker, 0.2);	
		//BasicColor = vec4(checker, checker, checker, 1.0);

		Roughness = 0.05;
	}
	else if( materialFator < BODY + 0.5 )
	{
		BasicColor = vec4(1.0, 0.541176, 0.639216, 1.0);
		Roughness = 0.8;
	}
	else if( materialFator < FOOT + 0.5)
	{
		BasicColor = vec4(1.0, 0.1941176, 0.189216, 1.0);
		Roughness = 0.2;
	}
	else if( materialFator < MOUTH + 0.5)
	{
		BasicColor = vec4(0.23137254901960784313725490196078, 0.0, 0.03529411764705882352941176470588, 1.0);
		Roughness = 0.5;
	}
	else if( materialFator < TONGUE + 0.5)
	{
		BasicColor = vec4(0.51372549019607843137254901960784, 0.03921568627450980392156862745098, 0.06274509803921568627450980392157, 1.0);
		Roughness = 0.8;
	}
	else if( materialFator < OUTER_EYE + 0.5)
	{
		float upDown = sin(u_TimeScreen.x * 0.00173) * 0.5;
	    vec3 diff =	 endPoint.xyz - vec3( 0.22, 0.25 - upDown, 0.95);

		vec3 blue = vec3(0.1137, 0.302, 0.851)* 2.0;
		vec3 black = vec3(0.0, 0.0, 0.0);

		if(diff.y > 0.0)
		{
			BasicColor = vec4(black , 1.0);
		}
		else
		{	
			float a = pow(abs(diff.y), 1.7) * 4.0;
			BasicColor = vec4(mix( black, blue, a) , 1.0);
		}

		Roughness = 1.0;
	}
	else if( materialFator < INNER_EYE + 0.5)
	{
		BasicColor = vec4(vec3(0.9), 1.0);
		Roughness = 0.7;
	}
	else if( materialFator < CHIN + 0.5)
	{
		BasicColor = vec4(1.0, 0.3941176, 0.389216, 1.0);
		Roughness = 0.8;
	}
    else if( materialFator < STAR_PANNEL + 0.5)
	{
		BasicColor = vec4(1.0, 1.0, 0.189216, 1.0);
		Roughness = 0.7;
	}
    else if( materialFator < STAR_BG + 0.5)
	{
		float res = 0.1;
		float timeSeed = u_TimeScreen.x * 0.01;
        float sinx = sin((endPoint.x + timeSeed) * res);
		float siny = sin((endPoint.y - timeSeed)* res);
        float cosx = cos((endPoint.x + timeSeed)* res);

		float rVal = (sinx + 1.0) * 0.5;
        float gVal = (cosx + 1.0) * 0.5;
	    float bVal = (sinx * siny + 1.0) * 0.5;

		BasicColor = vec4(rVal , gVal, bVal, 1.0);
		Roughness = 0.7;
	}

    BasicColor = clamp(BasicColor, 0.0, 1.0);
}

vec4 rainbow(float time, float resolution, float rainbowFreq, float rainbowSpeed, float rainbowWidth, float alphaFreq, float alphaSpeed, vec2 scale, vec2 offset, float cutoff )
{
    vec2 UV = fs_UV + offset;
    UV *= scale;
    float t = UV.y + sin(UV.x * rainbowFreq + time * rainbowSpeed) * rainbowWidth;

	t = floor(t * resolution) / 1.0;

    vec3 color = vec3(0.0);

    if(t > -0.5)
    {
        if(t< 0.5)
        {
            color = vec3(1.0, 0.0, 0.0);
        }
        else if(t< 1.5)
        {
            color = vec3(1.0, 0.5, 0.0);
        }
        else if(t< 2.5)
        {
            color = vec3(1.0, 1.0, 0.0);
        }
        else if(t< 3.5)
        {
            color = vec3(0.2, 1.0, 0.2);
        }
        else if(t< 4.5)
        {
            color = vec3(0.2, 0.2, 1.0);
        }
        else if(t< 5.5)
        {
            color = vec3(0.0, 0.1, 0.8);
        }
        else if(t< 6.5)
        {
            color = vec3(1.0, 0.0, 1.0);
        }
        else
        {
            return vec4(color, 0.0);
        }
    }
    else
    {
        return vec4(color, 0.0);
    }

    float alpha = clamp( cos(fs_UV.x * alphaFreq + time* alphaSpeed), 0.0, 1.0);
    alpha = clamp(alpha - cutoff, 0.0, 1.0);
    color = mix( vec3(0.0), color, alpha);

	return vec4(color, alpha);
}

void main() {
	// TODO: make a Raymarcher!
	out_Col = vec4(0.0, 0.0, 0.0, 1.0);

	vec3 eyePos = u_CameraPos.xyz;

	//get ViewDir from ScreenSpace
	
	vec3 viewVec = getViewVecFromScreenSpace(fs_UV, eyePos);
	vec3 lightVec = normalize(vec3(0.5, 1.0, 1.0));

	float epsilon = 2.0/(u_TimeScreen.w) * 0.25;

	bool isHit = false;

	vec3 normalVec;
	float AO = 0.0;
	float shadowFactor;
	vec4 reflectInfo;
	vec4 endPoint = rayMarching(viewVec, eyePos, lightVec, isHit, normalVec, epsilon, AO, shadowFactor, reflectInfo);

	float materialFator = endPoint.w;

		
	vec4 BasicColor = vec4(1.0, 0.5, 0.0, 1.0);
	float Roughness;

    getSurfaceColor(materialFator, endPoint.xyz, BasicColor, Roughness);

	
	
	vec4 SpecularColor = vec4(1.0);
	

	float NoL = dot(normalVec, lightVec);

	vec3 color = vec3(0.0);

	//Lighting - PBR
	if(NoL > 0.0 && shadowFactor > 0.0)
	{
		viewVec = -viewVec;
		
		float energyConservation = 1.0f - Roughness;

		vec3 specularTerm = vec3(0.0);

		float diffuseTerm = NoL;

		vec3 halfVec = viewVec + lightVec;
		halfVec = normalize(halfVec);
		float LoH = clamp(dot( lightVec, halfVec ), 0.0, 1.0);

		specularTerm = GGX_Spec(normalVec, halfVec, Roughness, BasicColor.xyz, SpecularColor.xyz, LightingFunGGX_FV(LoH, Roughness)) * energyConservation;

		specularTerm = clamp(specularTerm, 0.0, 2.0);

		color += (BasicColor.xyz + SpecularColor.xyz * specularTerm) * NoL;

        if(u_Factors01.y > 0.5 && Roughness < 0.79)
		{
			vec4 ReflectionColor;
			float refRoughness;
			getSurfaceColor(reflectInfo.w, reflectInfo.xyz, ReflectionColor, refRoughness);
			
			color += BasicColor.xyz * ReflectionColor.xyz * energyConservation * u_Factors01.z;
		}	

		//AO
		color *= AO;

		//shadow
		color *= shadowFactor;

			
	}
	
	if(isHit)
	{
		//Ambient
		color += BasicColor.xyz * 0.1;
		
	}
	else
	{
        // Rainbow
        vec4 rainbowColor = vec4(0.0, 0.0, 0.0, 0.0);

        rainbowColor = rainbow(u_TimeScreen.x, 25.0, 3.2, 0.01, 0.2, 3.0, 0.001, vec2(1.0, 1.0), vec2(0.0, -0.3), 0.0);
        rainbowColor = mix(rainbow(u_TimeScreen.x * 0.34, 64.0, 3.2, 0.01, 0.2, 3.0, 0.007, vec2(1.0, 1.0), vec2(0.0, -0.2), 0.2 ), rainbowColor, rainbowColor.w) ;
        rainbowColor = mix(rainbow(u_TimeScreen.x * 0.117, 128.0, 4.8, -0.01, 0.2, 3.0, -0.02, vec2(1.0, 1.0), vec2(0.0, -0.3), 0.6), rainbowColor, rainbowColor.w);
        color += rainbowColor.xyz;

		// Background stars
    	color += mix(addStars(u_TimeScreen.zw), rainbowColor.xyz, rainbowColor.w);
	}

	out_Col = vec4(color, 1.0);
}
