// WGSL source for the spectral path tracer. Built once per chapter instance with scene constants baked in.
//
//  pt_eye     – camera paths: hero-wavelength (4 λ) spectral path tracing, NEE + MIS to the ceiling lights,
//               Russian roulette, beam caustics gathered from the photon light-map at diffuse vertices.
//  pt_photon  – one step of a light (photon) path from the collimated beam: single wavelength, Sellmeier IOR,
//               exact dielectric Fresnel / TIR, deposits into the caustic light-map atlas and splats single
//               scattering from the haze straight onto the image (light tracing / camera connection).

import { ROOM, PLINTH, CEIL_LIGHTS, SPHERES, prismPlanes, brilliantPlanes } from './geometry.js';
import { normLines, LINES } from './spectral.js';

export const UNIFORMS = [ 'c0', 'c1', 'c2', 'c3', 'o0', 'o1', 'o2', 'g0', 'g1', 'b0', 'b1', 'b2', 'b3', 'a0', 'a1', 'm0', 's0' ];
export const LM_SCALE = 256.0;
export const SCR_SCALE = 65536.0;

const f = ( v ) => {

	const s = Number( v ).toPrecision( 9 );
	return /[.eE]/.test( s ) ? s : s + '.0';

};

const v3 = ( a ) => `vec3f(${f( a[ 0 ] )}, ${f( a[ 1 ] )}, ${f( a[ 2 ] )})`;

function convexFn( name, planes ) {

	let s = `fn ${name}(o: vec3f, d: vec3f) -> vec4f {
	var tn = -1e30; var tf = 1e30; var nn = vec3f(0.0); var nf = vec3f(0.0);
	var dn = 0.0; var ds = 0.0; var t = 0.0;
`;
	for ( const p of planes ) {

		const n = v3( p );
		s += `	dn = dot(${n}, d); ds = ${f( p[ 3 ] )} - dot(${n}, o);
	if (dn < -1e-9) { t = ds / dn; if (t > tn) { tn = t; nn = ${n}; } } else if (dn > 1e-9) { t = ds / dn; if (t < tf) { tf = t; nf = ${n}; } } else if (ds < 0.0) { tf = -1e30; }
`;

	}

	s += `	if (tn > tf || tf < 1e-5) { return vec4f(0.0, 0.0, 0.0, 1e20); }
	if (tn > 1e-5) { return vec4f(nn, tn); }
	return vec4f(nf, tf);
}
`;
	return s;

}

function linesFn( name, lines ) {

	const L = normLines( lines );
	const sum = L.reduce( ( a, [ , w ] ) => a + w, 0 );
	let acc = 0;
	let s = `fn ${name}(u: f32) -> vec2f {\n	let x = u * ${f( sum )};\n`;
	for ( let i = 0; i < L.length; i ++ ) {

		acc += L[ i ][ 1 ];
		s += i < L.length - 1 ? `	if (x < ${f( acc )}) { return vec2f(${f( L[ i ][ 0 ] )}, ${f( sum )}); }\n` : `	return vec2f(${f( L[ i ][ 0 ] )}, ${f( sum )});\n`;

	}

	return s + '}\n';

}

export function buildWGSL( atlas, opts ) {

	const R = ROOM;
	const gem = brilliantPlanes( 0.2 );
	const prism = prismPlanes();
	const uArgs = UNIFORMS.map( ( k ) => `${k}: vec4f` ).join( ', ' );
	const uPass = UNIFORMS.join( ', ' );

	// light-map surface mapping switch
	let lmCases = '';
	const W = R.x1 - R.x0, D = R.z1 - R.z0, H = R.y1;
	const pw = 2 * PLINTH.w, ph = PLINTH.h;
	const uvExpr = {
		1: `vec2f((p.x - ${f( R.x0 )}) / ${f( W )}, p.y / ${f( H )})`,
		2: `vec2f((p.z - ${f( R.z0 )}) / ${f( D )}, p.y / ${f( H )})`,
		3: `vec2f((p.z - ${f( R.z0 )}) / ${f( D )}, p.y / ${f( H )})`,
		4: `vec2f((p.x - ${f( R.x0 )}) / ${f( W )}, (p.z - ${f( R.z0 )}) / ${f( D )})`,
		5: `vec2f((p.x - ${f( R.x0 )}) / ${f( W )}, p.y / ${f( H )})`,
		6: `vec2f((p.x - ${f( R.x0 )}) / ${f( W )}, (p.z - ${f( R.z0 )}) / ${f( D )})`,
		7: `vec2f((p.x - S0.z + ${f( PLINTH.w )}) / ${f( pw )}, (p.z - S0.w + ${f( PLINTH.w )}) / ${f( pw )})`,
		8: `vec2f((p.z - S0.w + ${f( PLINTH.w )}) / ${f( pw )}, p.y / ${f( ph )})`,
		9: `vec2f((p.z - S0.w + ${f( PLINTH.w )}) / ${f( pw )}, p.y / ${f( ph )})`,
		10: `vec2f((p.x - S0.z + ${f( PLINTH.w )}) / ${f( pw )}, p.y / ${f( ph )})`,
		11: `vec2f((p.x - S0.z + ${f( PLINTH.w )}) / ${f( pw )}, p.y / ${f( ph )})`,
	};
	for ( const s of atlas.surf ) {

		lmCases += `		case ${s.id}u: { r.uv = ${uvExpr[ s.id ]}; r.rect = vec4f(${f( s.x )}, ${f( s.y )}, ${f( s.w )}, ${f( s.h )}); r.area = ${f( ( s.U / s.w ) * ( s.V / s.h ) )}; }\n`;

	}

	let lightHits = '', lightSel = '';
	CEIL_LIGHTS.forEach( ( L, k ) => {

		lightHits += `		{ let p = o + d * t; if (abs(p.x - ${f( L[ 0 ] )}) < ${f( L[ 2 ] )} && abs(p.z - ${f( L[ 1 ] )}) < ${f( L[ 3 ] )}) { (*h).t = t; (*h).n = vec3f(0.0, -1.0, 0.0); (*h).m = ${13 + k}u; } }\n`;
		lightSel += `		case ${k}u: { return vec4f(${f( L[ 0 ] )}, ${f( L[ 1 ] )}, ${f( L[ 2 ] )}, ${f( L[ 3 ] )}); }\n`;

	} );
	let lightScale = '';
	CEIL_LIGHTS.forEach( ( L, k ) => {

		lightScale += `		case ${13 + k}u: { return ${f( L[ 4 ] )}; }\n`;

	} );

	let sphereHits = '';
	for ( const S of SPHERES ) {

		sphereHits += `		pt_sphere(o, d, ${v3( S )}, ${f( S[ 3 ] )}, h);\n`;

	}

	const NL = CEIL_LIGHTS.length;
	const lib = /* wgsl */`
// ───────────── AETHER · Spectrum · spectral path tracer library ─────────────
const PT_PI: f32 = 3.14159265;
const PT_INV_PI: f32 = 0.318309886;
const PT_EPS: f32 = 2e-4;
const PT_BIG: f32 = 1e20;
const PT_RY: f32 = ${f( R.y1 )};
const PT_LM_W: f32 = ${f( atlas.W )};
const PT_LM_H: f32 = ${f( atlas.H )};
const PT_LM_SCALE: f32 = ${f( LM_SCALE )};
const PT_SCR_SCALE: f32 = ${f( SCR_SCALE )};
const PT_NL: u32 = ${NL}u;
const PT_MAXD: f32 = ${f( opts.maxDiffusePhoton )};
const PT_F0: f32 = 0.04;
const PT_PS: f32 = 0.5;

const M_BACK: u32 = 1u; const M_LEFT: u32 = 2u; const M_RIGHT: u32 = 3u; const M_CEIL: u32 = 4u; const M_FRONT: u32 = 5u;
const M_FLOOR: u32 = 6u; const M_PL_TOP: u32 = 7u; const M_PL_PX: u32 = 8u; const M_PL_NX: u32 = 9u; const M_PL_PZ: u32 = 10u; const M_PL_NZ: u32 = 11u;
const M_GLASS: u32 = 12u; const M_L0: u32 = 13u; const M_SLIT: u32 = 20u;

struct PtHit { t: f32, n: vec3f, m: u32, }
struct PtLm { uv: vec2f, rect: vec4f, area: f32, }

var<private> C0: vec4f; var<private> C1: vec4f; var<private> C2: vec4f; var<private> C3: vec4f;
var<private> O0: vec4f; var<private> O1: vec4f; var<private> O2: vec4f;
var<private> G0: vec4f; var<private> G1: vec4f;
var<private> B0: vec4f; var<private> B1: vec4f; var<private> B2: vec4f; var<private> B3: vec4f;
var<private> A0: vec4f; var<private> A1: vec4f; var<private> M0: vec4f; var<private> S0: vec4f;
var<private> pt_rng: u32;

fn pt_setup(${uArgs}) {
	C0 = c0; C1 = c1; C2 = c2; C3 = c3; O0 = o0; O1 = o1; O2 = o2; G0 = g0; G1 = g1;
	B0 = b0; B1 = b1; B2 = b2; B3 = b3; A0 = a0; A1 = a1; M0 = m0; S0 = s0;
}

fn pt_pcg(v: u32) -> u32 {
	let s = v * 747796405u + 2891336453u;
	let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (w >> 22u) ^ w;
}
fn pt_rand() -> f32 { pt_rng = pt_pcg(pt_rng); return f32(pt_rng >> 8u) * 5.96046448e-8; }

// ── spectral ──
fn pt_lobe(x: f32, mu: f32, s1: f32, s2: f32) -> f32 { let t = (x - mu) / select(s2, s1, x < mu); return exp(-0.5 * t * t); }
fn pt_cmf(l: f32) -> vec3f {
	let x = 1.056 * pt_lobe(l, 599.8, 37.9, 31.0) + 0.362 * pt_lobe(l, 442.0, 16.0, 26.7) - 0.065 * pt_lobe(l, 501.1, 20.4, 26.2);
	let y = 0.821 * pt_lobe(l, 568.8, 46.9, 40.5) + 0.286 * pt_lobe(l, 530.9, 16.3, 31.1);
	let z = 1.217 * pt_lobe(l, 437.0, 11.8, 36.0) + 0.681 * pt_lobe(l, 459.0, 26.0, 13.8);
	return vec3f(max(x, 0.0), y, z);
}
fn pt_xyz2rgb(c: vec3f) -> vec3f {
	return vec3f(3.2404542 * c.x - 1.5371385 * c.y - 0.4985314 * c.z,
		-0.9692660 * c.x + 1.8760108 * c.y + 0.0415560 * c.z,
		0.0556434 * c.x - 0.2040259 * c.y + 1.0572252 * c.z);
}
fn pt_planck(l: f32, T: f32) -> f32 { let x = l * 1e-3; return 1.0 / (x * x * x * x * x * (exp(14387.77 / (x * T)) - 1.0)); }
// visible-wavelength importance sampling (pdf ∝ sech²(0.0072 (λ − 538))) restricted to 380–720 nm
fn pt_lambda(u: f32) -> vec2f {
	let ta = -0.813603; let tb = 0.864326;
	let t = clamp(ta + u * (tb - ta), -0.99999, 0.99999);
	let l = 538.0 + atanh(t) / 0.0072;
	let c = cosh(0.0072 * (l - 538.0));
	return vec2f(l, 0.0072 / ((tb - ta) * c * c));
}
${linesFn( 'pt_lines_h', LINES.hydrogen )}
${linesFn( 'pt_lines_hg', LINES.mercury )}
// beam emission: returns (λ, Le(λ)/pdf(λ)) with Le normalised to luminance 1
fn pt_beam_lambda(u: f32, u2: f32) -> vec2f {
	let mode = u32(B2.w + 0.5);
	if (mode == 0u) {
		let ls = pt_lambda(u);
		return vec2f(ls.x, pt_planck(ls.x, M0.w) * B3.x / ls.y);
	}
	var r = pt_lines_h(u);
	if (mode == 2u) { r = pt_lines_hg(u); }
	// thermal/pressure broadening ~0.5 nm (Box-Muller-ish)
	r.x += 0.5 * sqrt(-2.0 * log(max(u2, 1e-6))) * cos(6.2831853 * fract(u * 7.31 + u2 * 3.17));
	return r;
}
// Sellmeier IOR with adjustable dispersion around n_d
fn pt_ior(l: f32) -> f32 {
	let x2 = l * l * 1e-6;
	let n2 = 1.0 + G0.x * x2 / (x2 - G1.x) + G0.y * x2 / (x2 - G1.y) + G0.z * x2 / (x2 - G1.z);
	return G1.w + (sqrt(n2) - G1.w) * G0.w;
}
fn pt_fresnel(cosi: f32, eta: f32) -> f32 {
	let s2 = (1.0 - cosi * cosi) / (eta * eta);
	if (s2 >= 1.0) { return 1.0; }
	let ct = sqrt(1.0 - s2);
	let rs = (cosi - eta * ct) / (cosi + eta * ct);
	let rp = (eta * cosi - ct) / (eta * cosi + ct);
	return 0.5 * (rs * rs + rp * rp);
}
fn pt_schlick(c: f32) -> f32 { let m = 1.0 - c; return PT_F0 + (1.0 - PT_F0) * m * m * m * m * m; }
fn pt_hg(c: f32, g: f32) -> f32 { let d = 1.0 + g * g - 2.0 * g * c; return (1.0 - g * g) / (4.0 * PT_PI * d * sqrt(d)); }

// ── geometry ──
fn pt_room(o: vec3f, d: vec3f, h: ptr<function, PtHit>) {
	if (d.x < 0.0) { let t = (${f( R.x0 )} - o.x) / d.x; if (t > 0.0 && t < (*h).t) { (*h).t = t; (*h).n = vec3f(1.0, 0.0, 0.0); (*h).m = M_LEFT; } }
	else if (d.x > 0.0) { let t = (${f( R.x1 )} - o.x) / d.x; if (t > 0.0 && t < (*h).t) { (*h).t = t; (*h).n = vec3f(-1.0, 0.0, 0.0); (*h).m = M_RIGHT; } }
	if (d.y < 0.0) { let t = -o.y / d.y; if (t > 0.0 && t < (*h).t) { (*h).t = t; (*h).n = vec3f(0.0, 1.0, 0.0); (*h).m = M_FLOOR; } }
	else if (d.y > 0.0) { let t = (PT_RY - o.y) / d.y; if (t > 0.0 && t < (*h).t) { (*h).t = t; (*h).n = vec3f(0.0, -1.0, 0.0); (*h).m = M_CEIL; } }
	if (d.z < 0.0) { let t = (${f( R.z0 )} - o.z) / d.z; if (t > 0.0 && t < (*h).t) { (*h).t = t; (*h).n = vec3f(0.0, 0.0, 1.0); (*h).m = M_BACK; } }
	else if (d.z > 0.0) { let t = (${f( R.z1 )} - o.z) / d.z; if (t > 0.0 && t < (*h).t) { (*h).t = t; (*h).n = vec3f(0.0, 0.0, -1.0); (*h).m = M_FRONT; } }
}

fn pt_plinth(o: vec3f, d: vec3f, h: ptr<function, PtHit>) {
	if (S0.y < 0.5) { return; }
	let bmin = vec3f(S0.z - ${f( PLINTH.w )}, -1.0, S0.w - ${f( PLINTH.w )});
	let bmax = vec3f(S0.z + ${f( PLINTH.w )}, ${f( PLINTH.h )}, S0.w + ${f( PLINTH.w )});
	let inv = vec3f(select(1e30, 1.0 / d.x, abs(d.x) > 1e-12), select(1e30, 1.0 / d.y, abs(d.y) > 1e-12), select(1e30, 1.0 / d.z, abs(d.z) > 1e-12));
	let t0 = (bmin - o) * inv; let t1 = (bmax - o) * inv;
	let tmn = min(t0, t1); let tmx = max(t0, t1);
	let tn = max(max(tmn.x, tmn.y), tmn.z); let tf = min(min(tmx.x, tmx.y), tmx.z);
	if (tn > tf || tn <= 1e-5 || tn >= (*h).t) { return; }
	(*h).t = tn;
	if (tn == tmn.y) { (*h).n = vec3f(0.0, 1.0, 0.0); (*h).m = M_PL_TOP; }
	else if (tn == tmn.x) { if (d.x > 0.0) { (*h).n = vec3f(-1.0, 0.0, 0.0); (*h).m = M_PL_NX; } else { (*h).n = vec3f(1.0, 0.0, 0.0); (*h).m = M_PL_PX; } }
	else { if (d.z > 0.0) { (*h).n = vec3f(0.0, 0.0, -1.0); (*h).m = M_PL_NZ; } else { (*h).n = vec3f(0.0, 0.0, 1.0); (*h).m = M_PL_PZ; } }
}

${convexFn( 'pt_prism_hit', prism )}
${convexFn( 'pt_gem_hit', gem.planes )}

fn pt_sphere(o: vec3f, d: vec3f, c: vec3f, r: f32, h: ptr<function, PtHit>) {
	let oc = o - c; let b = dot(oc, d); let cc = dot(oc, oc) - r * r;
	let disc = b * b - cc;
	if (disc <= 0.0) { return; }
	let s = sqrt(disc);
	var t = -b - s;
	if (t < 1e-5) { t = -b + s; }
	if (t < 1e-5 || t >= (*h).t) { return; }
	(*h).t = t; (*h).n = (oc + d * t) / r; (*h).m = M_GLASS;
}

fn pt_glass(o: vec3f, d: vec3f, h: ptr<function, PtHit>) {
	let preset = u32(S0.x + 0.5);
	if (preset == 2u) {
${sphereHits}		return;
	}
	let pos = vec3f(O0.w, O1.w, O2.w);
	let lo = o - pos;
	if (preset == 1u) {
		let b = dot(lo, d); let cc = dot(lo, lo) - ${f( gem.radius * gem.radius )};
		if (cc > 0.0 && (b > 0.0 || b * b - cc < 0.0)) { return; }
	}
	let ol = O0.xyz * lo.x + O1.xyz * lo.y + O2.xyz * lo.z;
	let dl = O0.xyz * d.x + O1.xyz * d.y + O2.xyz * d.z;
	var r: vec4f;
	if (preset == 0u) { r = pt_prism_hit(ol, dl); } else { r = pt_gem_hit(ol, dl); }
	if (r.w < (*h).t) {
		(*h).t = r.w; (*h).m = M_GLASS;
		(*h).n = vec3f(dot(O0.xyz, r.xyz), dot(O1.xyz, r.xyz), dot(O2.xyz, r.xyz));
	}
}

fn pt_lights(o: vec3f, d: vec3f, h: ptr<function, PtHit>) {
	if (d.y > 0.0) {
		let t = (PT_RY - 0.004 - o.y) / d.y;
		if (t > 0.0 && t < (*h).t) {
${lightHits}		}
	}
	// beam aperture (slit) – emits along +B1
	let dd = dot(d, B1.xyz);
	if (dd < -1e-6) {
		let t = dot(B0.xyz - o, B1.xyz) / dd;
		if (t > 0.0 && t < (*h).t) {
			let q = o + d * t - B0.xyz;
			let V = cross(B1.xyz, B2.xyz);
			if (abs(dot(q, B2.xyz)) < B0.w * 1.15 && abs(dot(q, V)) < B1.w * 1.05) { (*h).t = t; (*h).n = B1.xyz; (*h).m = M_SLIT; }
		}
	}
}

fn pt_scene(o: vec3f, d: vec3f, emit: bool) -> PtHit {
	var h = PtHit(PT_BIG, vec3f(0.0, 1.0, 0.0), 0u);
	pt_room(o, d, &h);
	pt_plinth(o, d, &h);
	pt_glass(o, d, &h);
	if (emit) { pt_lights(o, d, &h); }
	return h;
}

fn pt_occluded(o: vec3f, d: vec3f, tmax: f32) -> bool {
	var h = PtHit(tmax, vec3f(0.0), 0u);
	pt_plinth(o, d, &h);
	pt_glass(o, d, &h);
	return h.m != 0u;
}

fn pt_albedo(m: u32) -> f32 {
	switch m {
		case 1u: { return 0.75; }
		case 2u, 3u: { return 0.42; }
		case 4u, 5u: { return 0.32; }
		case 6u: { return A1.z; }
		case 7u: { return 0.5; }
		default: { return 0.06; }
	}
}

fn pt_light_rect(k: u32) -> vec4f {
	switch k {
${lightSel}		default: { return vec4f(0.0); }
	}
}
fn pt_light_scale(m: u32) -> f32 {
	switch m {
${lightScale}		default: { return 0.0; }
	}
}

// ── caustic light-map atlas ──
fn pt_lm_map(m: u32, p: vec3f) -> PtLm {
	var r = PtLm(vec2f(0.0), vec4f(0.0), 0.0);
	switch m {
${lmCases}		default: { r.area = -1.0; }
	}
	return r;
}
fn pt_lm(lm: texture_2d<f32>, lms: sampler, m: u32, p: vec3f) -> vec3f {
	if (m == 0u || m > 11u) { return vec3f(0.0); }
	if (m >= 7u && S0.y < 0.5) { return vec3f(0.0); }
	let L = pt_lm_map(m, p);
	let tc = L.rect.xy + clamp(L.uv * L.rect.zw, vec2f(0.5), L.rect.zw - vec2f(0.5));
	return textureSampleLevel(lm, lms, tc / vec2f(PT_LM_W, PT_LM_H), 0.0).xyz;
}
fn pt_lm_index(m: u32, p: vec3f) -> vec2f {
	if (m == 0u || m > 11u) { return vec2f(-1.0, 0.0); }
	let L = pt_lm_map(m, p);
	let tx = clamp(floor(L.uv.x * L.rect.z), 0.0, L.rect.z - 1.0);
	let ty = clamp(floor(L.uv.y * L.rect.w), 0.0, L.rect.w - 1.0);
	return vec2f((L.rect.y + ty) * PT_LM_W + L.rect.x + tx, L.area);
}

// ── BSDF helpers ──
fn pt_onb(n: vec3f) -> mat3x3f {
	let s = select(-1.0, 1.0, n.z >= 0.0);
	let a = -1.0 / (s + n.z); let b = n.x * n.y * a;
	return mat3x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}
fn pt_cosine(n: vec3f, u1: f32, u2: f32) -> vec3f {
	let r = sqrt(u1); let ph = 6.2831853 * u2;
	return normalize(pt_onb(n) * vec3f(r * cos(ph), r * sin(ph), sqrt(max(0.0, 1.0 - u1))));
}
fn pt_g1(c: f32, a2: f32) -> f32 { return 2.0 * c / (c + sqrt(a2 + (1.0 - a2) * c * c)); }
// returns (f, pdf of VNDF sampling) for the GGX coat
fn pt_ggx(n: vec3f, wo: vec3f, wi: vec3f, a: f32) -> vec2f {
	let nv = dot(n, wo); let nl = dot(n, wi);
	if (nv <= 0.0 || nl <= 0.0) { return vec2f(0.0); }
	let hv = normalize(wo + wi); let nh = max(dot(n, hv), 0.0); let vh = max(dot(wo, hv), 0.0);
	let a2 = a * a;
	let dd = nh * nh * (a2 - 1.0) + 1.0;
	let D = a2 / (PT_PI * dd * dd);
	let g1v = pt_g1(nv, a2);
	let F = pt_schlick(vh);
	return vec2f(D * g1v * pt_g1(nl, a2) * F / (4.0 * nv * nl), g1v * D / (4.0 * nv));
}
fn pt_ggx_sample(n: vec3f, wo: vec3f, a: f32, u1: f32, u2: f32) -> vec3f {
	let tb = pt_onb(n);
	let v = vec3f(dot(wo, tb[0]), dot(wo, tb[1]), dot(wo, tb[2]));
	let vh = normalize(vec3f(a * v.x, a * v.y, v.z));
	let ls = vh.x * vh.x + vh.y * vh.y;
	let T1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-vh.y, vh.x, 0.0) / sqrt(ls), ls > 0.0);
	let T2 = cross(vh, T1);
	let r = sqrt(u1); let ph = 6.2831853 * u2;
	let t1 = r * cos(ph); let s = 0.5 * (1.0 + vh.z);
	let t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * r * sin(ph);
	let nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * vh;
	let ne = normalize(vec3f(a * nh.x, a * nh.y, max(1e-6, nh.z)));
	let hw = tb * ne;
	return reflect(-wo, hw);
}
// dielectric interface: returns (new direction, +1 reflected / -1 transmitted)
fn pt_dielectric(d: vec3f, n: vec3f, ior: f32, u: f32) -> vec4f {
	let entering = dot(d, n) < 0.0;
	let nn = select(-n, n, entering);
	let eta = select(1.0 / ior, ior, entering);
	let ci = -dot(d, nn);
	let F = pt_fresnel(ci, eta);
	if (u < F) { return vec4f(reflect(d, nn), 1.0); }
	return vec4f(normalize(refract(d, nn, 1.0 / eta)), -1.0);
}

// ── light tracing: splat single scattering of the beam haze onto the image ──
fn pt_cam_splat(m: vec3f, wd: vec3f, val: f32, lam: f32) -> vec4f {
	let toc = C0.xyz - m; let dist = length(toc); let wc = toc / dist;
	let v = -wc;
	let cz = dot(v, C3.xyz);
	if (cz < 1e-3) { return vec4f(-1.0); }
	let sx = dot(v, C1.xyz) / (cz * C1.w); let sy = dot(v, C2.xyz) / (cz * C0.w);
	if (abs(sx) >= 1.0 || abs(sy) >= 1.0) { return vec4f(-1.0); }
	if (pt_occluded(m, wc, dist)) { return vec4f(-1.0); }
	let W = C2.w; let H = C3.w;
	let px = min(floor((sx * 0.5 + 0.5) * W), W - 1.0);
	let py = min(floor((0.5 - sy * 0.5) * H), H - 1.0);
	let omega = 4.0 * C0.w * C0.w / (H * H) * cz * cz * cz;
	let rad = val * pt_hg(dot(wd, wc), B3.w) / (dist * dist * omega);
	let c = min(pt_cmf(lam) * (rad * PT_SCR_SCALE), vec3f(1e9));
	return vec4f(py * W + px, floor(c + vec3f(pt_rand(), pt_rand(), pt_rand())));
}

// ── camera path ──
fn pt_amb_w(lam: vec4f, tw: vec4f) -> vec3f {
	var s = vec3f(0.0);
	for (var i = 0; i < 4; i++) {
		if (tw[i] > 0.0) { s += tw[i] * pt_planck(lam[i], A1.x) * A1.y * pt_cmf(lam[i]); }
	}
	return s;
}

fn pt_path(ro0: vec3f, rd0: vec3f, uw: f32, first: bool, gd: ptr<function, vec4f>, gp: ptr<function, vec4f>, lm: texture_2d<f32>, lms: sampler) -> vec3f {
	var lam = vec4f(0.0); var tw = vec4f(0.0);
	for (var i = 0; i < 4; i++) {
		let ls = pt_lambda(fract(uw + f32(i) * 0.25));
		lam[i] = ls.x; tw[i] = 0.25 / ls.y;
	}
	var wamb = pt_amb_w(lam, tw);
	var ro = ro0; var rd = rd0;
	var Aa = 1.0; var Ab = 1.0;
	var specPrev = true; var pdfPrev = 1.0; var single = false;
	var xyz = vec3f(0.0);
	let maxB = i32(A0.y);
	let alpha = A1.w;
	for (var b = 0; b < maxB; b++) {
		let h = pt_scene(ro, rd, true);
		if (h.t >= PT_BIG) { break; }
		let p = ro + rd * h.t;
		if (first && b == 0) { *gd = vec4f(h.n, h.t); *gp = vec4f(p, f32(h.m)); }
		if (h.m >= M_L0 && h.m < M_L0 + PT_NL) {
			var w = 1.0;
			if (!specPrev) {
				let rc = pt_light_rect(h.m - M_L0);
				let pdfL = h.t * h.t / (max(rd.y, 1e-4) * 4.0 * rc.z * rc.w * f32(PT_NL));
				w = pdfPrev * pdfPrev / (pdfPrev * pdfPrev + pdfL * pdfL);
			}
			xyz += (Aa * w * A0.x * pt_light_scale(h.m)) * wamb;
			break;
		}
		if (h.m == M_SLIT) { xyz += Ab * M0.xyz; break; }
		if (h.m == M_GLASS) {
			if (!single) { single = true; tw = vec4f(tw.x * 4.0, 0.0, 0.0, 0.0); wamb = pt_amb_w(lam, tw); }
			let r = pt_dielectric(rd, h.n, pt_ior(lam.x), pt_rand());
			let nn = select(-h.n, h.n, dot(rd, h.n) < 0.0);
			ro = p + nn * select(-PT_EPS, PT_EPS, r.w > 0.0);
			rd = r.xyz;
			specPrev = true;
			continue;
		}
		let n = h.n; let wo = -rd;
		let isFloor = h.m == M_FLOOR;
		var kd = pt_albedo(h.m);
		if (isFloor) { kd *= 1.0 - pt_schlick(max(dot(n, wo), 0.0)); }
		// beam caustics: photon light-map (all photon bounces) through the diffuse lobe
		if (Ab > 0.0) { xyz += (Ab * kd * PT_INV_PI) * pt_lm(lm, lms, h.m, p); }
		let po = p + n * PT_EPS;
		// next-event estimation of the ceiling lights (MIS, power heuristic)
		{
			let k = min(u32(pt_rand() * f32(PT_NL)), PT_NL - 1u);
			let rc = pt_light_rect(k);
			let q = vec3f(rc.x + (pt_rand() * 2.0 - 1.0) * rc.z, PT_RY - 0.004, rc.y + (pt_rand() * 2.0 - 1.0) * rc.w);
			var wi = q - po; let d2 = dot(wi, wi); let dist = sqrt(d2); wi /= dist;
			let cl = wi.y; let ci = dot(n, wi);
			if (cl > 0.0 && ci > 0.0) {
				if (!pt_occluded(po, wi, dist - 1e-3)) {
					let pdfL = d2 / (cl * 4.0 * rc.z * rc.w * f32(PT_NL));
					var fb = kd * PT_INV_PI; var pdfB = ci * PT_INV_PI;
					if (isFloor) { let g = pt_ggx(n, wo, wi, alpha); fb += g.x; pdfB = PT_PS * g.y + (1.0 - PT_PS) * pdfB; }
					let w = pdfL * pdfL / (pdfL * pdfL + pdfB * pdfB);
					xyz += (Aa * fb * ci / pdfL * w * A0.x * pt_light_scale(M_L0 + k)) * wamb;
				}
			}
		}
		// BSDF sampling
		var wi: vec3f;
		if (isFloor) {
			if (pt_rand() < PT_PS) { wi = pt_ggx_sample(n, wo, alpha, pt_rand(), pt_rand()); } else { wi = pt_cosine(n, pt_rand(), pt_rand()); }
			let ci = dot(n, wi);
			if (ci <= 0.0) { break; }
			let g = pt_ggx(n, wo, wi, alpha);
			let pdf = PT_PS * g.y + (1.0 - PT_PS) * ci * PT_INV_PI;
			if (pdf <= 0.0) { break; }
			Aa *= (g.x + kd * PT_INV_PI) * ci / pdf;
			Ab *= g.x * ci / pdf;
			pdfPrev = pdf;
		} else {
			wi = pt_cosine(n, pt_rand(), pt_rand());
			pdfPrev = max(dot(n, wi), 1e-6) * PT_INV_PI;
			Aa *= kd; Ab = 0.0;
		}
		specPrev = false;
		ro = po; rd = wi;
		if (b >= 2) {
			let q = clamp(max(Aa, Ab), 0.05, 0.95);
			if (pt_rand() > q) { break; }
			Aa /= q; Ab /= q;
		}
	}
	return xyz;
}
`;

	const eye = /* wgsl */`fn pt_eye(pix: vec2f, nidx: u32, seed: u32, gd: ptr<function, vec4f>, gp: ptr<function, vec4f>, lm: texture_2d<f32>, lms: sampler, ${uArgs}) -> vec4f {
	pt_setup(${uPass});
	pt_rng = pt_pcg(seed);
	let spp = max(u32(A0.z), 1u);
	var acc = vec3f(0.0);
	let hp = pt_pcg(u32(pix.x) * 1973u + u32(pix.y) * 9277u + 26699u);
	let ign = fract(52.9829189 * fract(0.06711056 * pix.x + 0.00583715 * pix.y));
	for (var s = 0u; s < spp; s++) {
		let n = nidx * spp + s;
		// R2 low-discrepancy sequence, Cranley-Patterson rotated per pixel; IGN-rotated golden-ratio wavelengths
		let j = fract(vec2f(f32(hp & 65535u), f32(hp >> 16u)) / 65536.0 + vec2f(f32(n * 3242174889u), f32(n * 2447445413u)) * 2.3283064e-10);
		let uw = fract(ign + f32(n * 2654435769u) * 2.3283064e-10);
		let sx = ((pix.x + j.x) / C2.w) * 2.0 - 1.0;
		let sy = 1.0 - ((pix.y + j.y) / C3.w) * 2.0;
		let rd = normalize(C3.xyz + C1.xyz * (sx * C1.w) + C2.xyz * (sy * C0.w));
		acc += pt_path(C0.xyz, rd, uw, s == 0u, gd, gp, lm, lms);
	}
	return vec4f(pt_xyz2rgb(acc / f32(spp)), 1.0);
}
`;

	const photon = /* wgsl */`fn pt_photon(ro: ptr<function, vec4f>, rd: ptr<function, vec4f>, st: ptr<function, vec4f>, seed: ptr<function, u32>, spl: ptr<function, vec4f>, sps: ptr<function, vec4f>, ${uArgs}) -> f32 {
	pt_setup(${uPass});
	*spl = vec4f(-1.0); *sps = vec4f(-1.0);
	if ((*st).z < 0.5) { return 0.0; }
	if ((*st).y < 0.5) {
		pt_rng = pt_pcg(*seed ^ 0x9E3779B9u);
		let u1 = pt_rand() * 2.0 - 1.0; let u2 = pt_rand() * 2.0 - 1.0;
		let V = cross(B1.xyz, B2.xyz);
		let org = B0.xyz + B2.xyz * (u1 * B0.w) + V * (u2 * B1.w);
		let ls = pt_beam_lambda(pt_rand(), pt_rand());
		*ro = vec4f(org, ls.x); *rd = vec4f(B1.xyz, ls.y * B3.y); *st = vec4f(0.0, 1.0, 1.0, 1.0);
	} else {
		pt_rng = *seed;
	}
	let o = (*ro).xyz; let d = (*rd).xyz; let lam = (*ro).w; let T = (*rd).w;
	let h = pt_scene(o, d, false);
	if ((*st).w > 0.5 && B3.z > 0.0) {
		let L = min(h.t, 14.0);
		*sps = pt_cam_splat(o + d * (pt_rand() * L), d, T * B3.z * L, lam);
	}
	if (h.t >= PT_BIG) { (*st).z = 0.0; *seed = pt_rng; return 0.0; }
	let p = o + d * h.t;
	(*st).y += 1.0;
	if (h.m == M_GLASS) {
		let r = pt_dielectric(d, h.n, pt_ior(lam), pt_rand());
		let nn = select(-h.n, h.n, dot(d, h.n) < 0.0);
		*ro = vec4f(p + nn * select(-PT_EPS, PT_EPS, r.w > 0.0), lam);
		*rd = vec4f(r.xyz, T);
		*seed = pt_rng; return 1.0;
	}
	let li = pt_lm_index(h.m, p);
	if (li.x >= 0.0 && li.y > 0.0) {
		let c = min(pt_cmf(lam) * (T / li.y * PT_LM_SCALE), vec3f(1e9));
		*spl = vec4f(li.x, floor(c + vec3f(pt_rand(), pt_rand(), pt_rand())));
	}
	let n = h.n; let wo = -d;
	var wi = vec3f(0.0);
	var beam = 0.0;
	var Tn = T;
	let alb = pt_albedo(h.m);
	if (h.m == M_FLOOR && pt_rand() < pt_schlick(max(dot(n, wo), 0.0))) {
		wi = pt_ggx_sample(n, wo, A1.w, pt_rand(), pt_rand());
		let cl = dot(n, wi);
		if (cl <= 0.0) { (*st).z = 0.0; *seed = pt_rng; return 0.0; }
		Tn *= pt_g1(cl, A1.w * A1.w);
		beam = (*st).w;
	} else {
		(*st).x += 1.0;
		if ((*st).x > PT_MAXD || pt_rand() > alb) { (*st).z = 0.0; *seed = pt_rng; return 0.0; }
		wi = pt_cosine(n, pt_rand(), pt_rand());
	}
	*ro = vec4f(p + n * PT_EPS, lam); *rd = vec4f(wi, Tn); (*st).w = beam;
	*seed = pt_rng;
	return 1.0;
}
`;

	return { lib, eye, photon, gemRadius: gem.radius };

}
