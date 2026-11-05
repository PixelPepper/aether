// Generative ambient score. Pure WebAudio, no samples.
// One "mood" per chapter: a slowly breathing pad chord, a filtered-noise wind bed and sparse shimmering bells.
// Moods crossfade on chapter change; pointer pressure swells the pad filter.

const MOODS = {
	genesis: { root: 49.0, chord: [ 0, 7, 12, 15, 19 ], cutoff: 900, bells: 0.55, wind: 0.10, scale: [ 0, 3, 7, 10, 12, 15 ] },
	nebula: { root: 43.65, chord: [ 0, 7, 14, 16, 21 ], cutoff: 700, bells: 0.35, wind: 0.22, scale: [ 0, 2, 7, 9, 14 ] },
	collision: { root: 36.71, chord: [ 0, 5, 12, 17, 22 ], cutoff: 520, bells: 0.25, wind: 0.30, scale: [ 0, 5, 7, 10, 12 ] },
	ocean: { root: 41.2, chord: [ 0, 7, 12, 16, 23 ], cutoff: 650, bells: 0.45, wind: 0.26, scale: [ 0, 4, 7, 11, 14 ] },
	horizon: { root: 32.7, chord: [ 0, 7, 12, 13, 19 ], cutoff: 380, bells: 0.18, wind: 0.18, scale: [ 0, 1, 7, 8, 12 ] },
};
const DEFAULT = MOODS.genesis;

export class Score {

	constructor() {

		this.ctx = null;
		this.enabled = false;
		this.mood = DEFAULT;
		this.voices = [];
		this.press = 0;

	}

	async start() {

		if ( this.ctx ) {

			await this.ctx.resume();
			this.enabled = true;
			this.master.gain.setTargetAtTime( 0.9, this.ctx.currentTime, 0.8 );
			return;

		}

		const ac = this.ctx = new AudioContext();
		this.enabled = true;

		this.master = ac.createGain();
		this.master.gain.value = 0;
		const comp = ac.createDynamicsCompressor();
		comp.threshold.value = - 18; comp.ratio.value = 3;
		this.master.connect( comp ).connect( ac.destination );

		// algorithmic reverb impulse: exponentially decaying stereo noise with a slow bloom
		this.verb = ac.createConvolver();
		this.verb.buffer = this._impulse( 7.5 );
		const wet = ac.createGain(); wet.gain.value = 0.85;
		this.verb.connect( wet ).connect( this.master );

		this.dry = ac.createGain(); this.dry.gain.value = 0.35;
		this.dry.connect( this.master );

		// pad bus with a resonant low-pass that breathes
		this.padFilter = ac.createBiquadFilter();
		this.padFilter.type = 'lowpass'; this.padFilter.Q.value = 0.7; this.padFilter.frequency.value = 600;
		this.padGain = ac.createGain(); this.padGain.gain.value = 0.16;
		this.padFilter.connect( this.padGain );
		this.padGain.connect( this.dry ); this.padGain.connect( this.verb );

		const lfo = ac.createOscillator(); lfo.frequency.value = 0.045;
		const lfoAmt = ac.createGain(); lfoAmt.gain.value = 220;
		lfo.connect( lfoAmt ).connect( this.padFilter.frequency ); lfo.start();

		// wind: pink-ish noise through a wandering band-pass
		const noise = ac.createBufferSource();
		noise.buffer = this._noise( 6 ); noise.loop = true;
		this.windFilter = ac.createBiquadFilter(); this.windFilter.type = 'bandpass'; this.windFilter.Q.value = 0.9; this.windFilter.frequency.value = 500;
		this.windGain = ac.createGain(); this.windGain.gain.value = 0;
		noise.connect( this.windFilter ).connect( this.windGain );
		this.windGain.connect( this.verb ); this.windGain.connect( this.dry );
		noise.start();
		const wl = ac.createOscillator(); wl.frequency.value = 0.031;
		const wla = ac.createGain(); wla.gain.value = 380;
		wl.connect( wla ).connect( this.windFilter.frequency ); wl.start();

		this._applyMood( this.mood, 0.01 );
		this.master.gain.setTargetAtTime( 0.9, ac.currentTime, 1.5 );
		this._bellLoop();

	}

	stop() {

		if ( ! this.ctx ) return;
		this.enabled = false;
		this.master.gain.setTargetAtTime( 0, this.ctx.currentTime, 0.4 );
		setTimeout( () => ! this.enabled && this.ctx.suspend(), 1500 );

	}

	setChapter( id ) {

		this.mood = MOODS[ id ] || DEFAULT;
		if ( this.ctx ) this._applyMood( this.mood, 2.5 );

	}

	// 0..1 interaction energy, called every frame by the shell
	setPress( v ) {

		if ( ! this.ctx || ! this.enabled ) return;
		this.press += ( v - this.press ) * 0.05;
		this.padFilter.Q.setTargetAtTime( 0.7 + this.press * 6, this.ctx.currentTime, 0.2 );
		this.padGain.gain.setTargetAtTime( 0.16 + this.press * 0.06, this.ctx.currentTime, 0.3 );

	}

	_applyMood( m, fade ) {

		const ac = this.ctx, t = ac.currentTime;
		for ( const v of this.voices ) {

			v.gain.gain.setTargetAtTime( 0, t, fade / 3 );
			v.oscs.forEach( ( o ) => o.stop( t + fade * 3 ) );

		}

		this.voices = m.chord.map( ( semi, i ) => {

			const f = m.root * Math.pow( 2, semi / 12 ) * 2;
			const gain = ac.createGain(); gain.gain.value = 0;
			const pan = ac.createStereoPanner(); pan.pan.value = ( i / ( m.chord.length - 1 ) ) * 1.2 - 0.6;
			const oscs = [ - 7, 0, 6 ].map( ( det, k ) => {

				const o = ac.createOscillator();
				o.type = k === 1 ? 'triangle' : 'sawtooth';
				o.frequency.value = f; o.detune.value = det + ( Math.random() - 0.5 ) * 4;
				const g = ac.createGain(); g.gain.value = k === 1 ? 0.5 : 0.18;
				o.connect( g ).connect( gain ); o.start();
				return o;

			} );
			// each voice swells independently so the chord never sits still
			const sw = ac.createOscillator(); sw.frequency.value = 0.03 + Math.random() * 0.05;
			const swa = ac.createGain(); swa.gain.value = 0.25;
			sw.connect( swa ).connect( gain.gain ); sw.start(); oscs.push( sw );
			gain.connect( pan ).connect( this.padFilter );
			gain.gain.setTargetAtTime( 0.4 / m.chord.length * 2, t + 0.05, fade / 2 );
			return { gain, oscs };

		} );

		this.padFilter.frequency.setTargetAtTime( m.cutoff, t, fade );
		this.windGain.gain.setTargetAtTime( m.wind, t, fade );

	}

	_bellLoop() {

		if ( ! this.ctx ) return;
		const m = this.mood;
		if ( this.enabled && Math.random() < m.bells ) this._bell( m );
		setTimeout( () => this._bellLoop(), 900 + Math.random() * 2600 );

	}

	_bell( m ) {

		const ac = this.ctx, t = ac.currentTime + 0.02;
		const semi = m.scale[ ( Math.random() * m.scale.length ) | 0 ] + 12 * ( 3 + ( Math.random() < 0.35 ? 1 : 0 ) );
		const f = m.root * Math.pow( 2, semi / 12 );
		const out = ac.createGain(); out.gain.value = 0;
		const pan = ac.createStereoPanner(); pan.pan.value = Math.random() * 1.6 - 0.8;
		out.connect( pan ); pan.connect( this.verb );
		const g2 = ac.createGain(); g2.gain.value = 0.25; pan.connect( g2 ).connect( this.dry );
		// FM bell: inharmonic modulator ratio
		const car = ac.createOscillator(); car.frequency.value = f;
		const mod = ac.createOscillator(); mod.frequency.value = f * 3.507;
		const modAmt = ac.createGain(); modAmt.gain.setValueAtTime( f * 1.6, t ); modAmt.gain.exponentialRampToValueAtTime( 1, t + 2.5 );
		mod.connect( modAmt ).connect( car.frequency );
		car.connect( out );
		const amp = 0.05 + Math.random() * 0.05;
		out.gain.setValueAtTime( 0, t ); out.gain.linearRampToValueAtTime( amp, t + 0.01 ); out.gain.exponentialRampToValueAtTime( 0.0001, t + 6 );
		car.start( t ); mod.start( t ); car.stop( t + 6.2 ); mod.stop( t + 6.2 );

	}

	_impulse( seconds ) {

		const ac = this.ctx, len = Math.floor( ac.sampleRate * seconds );
		const buf = ac.createBuffer( 2, len, ac.sampleRate );
		for ( let ch = 0; ch < 2; ch ++ ) {

			const d = buf.getChannelData( ch );
			for ( let i = 0; i < len; i ++ ) {

				const x = i / len;
				d[ i ] = ( Math.random() * 2 - 1 ) * Math.pow( 1 - x, 3.2 ) * Math.min( 1, x * 40 );

			}

		}

		return buf;

	}

	_noise( seconds ) {

		const ac = this.ctx, len = Math.floor( ac.sampleRate * seconds );
		const buf = ac.createBuffer( 2, len, ac.sampleRate );
		for ( let ch = 0; ch < 2; ch ++ ) {

			const d = buf.getChannelData( ch );
			let b0 = 0, b1 = 0, b2 = 0;
			for ( let i = 0; i < len; i ++ ) {

				const w = Math.random() * 2 - 1;
				b0 = 0.997 * b0 + w * 0.029; b1 = 0.985 * b1 + w * 0.032; b2 = 0.95 * b2 + w * 0.048;
				d[ i ] = ( b0 + b1 + b2 ) * 0.9;

			}

		}

		return buf;

	}

}
