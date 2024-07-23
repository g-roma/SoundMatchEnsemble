SynthFactory{
	/*
	This class implements the following template:

	                      MOD       MOD
	MOD -- OSC ----┐       |         |
	              MIX -- FILTER --- AMP -- OUT
	MOD -- OSC ----┘

	The choice of modules is specified by a 7-digit ID (as string) in the constructor
	where each digit selects from source1Classes, source2Classes, modClasses, filterClasses

	*/
	var <server;
	var func,  paramNames, specs;
	var pitch;
	var dur = 1.0;
	var src1Func, src2Func, mixFunc, filterFunc, ampFunc;
	var mod1Func, mod2Func, mod3Func, mod4Func;

	classvar source1Classes, source2Classes, modClasses, filterClasses;
	classvar nextAvailablePort;

	*initClass{
		source1Classes = [VarSaw, Pulse];
		source2Classes = [Saw, WhiteNoise];
		modClasses = [LFTri, EnvGen, DC];
		filterClasses = [RLPF, RHPF, BPF];
	}

	*new {|synthID|
		^super.new.init(synthID);
	}

	init{|synthID|
		var m1Class, m2Class, m3Class, m4Class, s1Class, s2Class, fClass;
		var digits = synthID.as(Array).collect{|x|x.digit};
		s1Class = source1Classes[digits[0]];
		s2Class = source2Classes[digits[1]];
		m1Class = modClasses[digits[2]];
		m2Class = modClasses[digits[3]];
		m3Class = modClasses[digits[4]];
		m4Class = modClasses[digits[5]];
		fClass = filterClasses[digits[6]];

		server = Server.local;

		pitch = 60.midicps;
		specs = Dictionary.new;
		mod1Func = this.addModulator(m1Class);
		mod2Func = this.addModulator(m2Class);
		mod3Func = this.addModulator(m3Class);
		mod4Func = this.addModulator(m4Class);
		src1Func = this.addSource(s1Class);
		src2Func = this.addSource(s2Class);
		mixFunc = this.addMix;
		filterFunc = this.addFilter(fClass);
		ampFunc = this.addAmp;
	}

	randomParams{
		var funcs = [mod1Func, mod2Func, mod3Func, mod4Func,
			src1Func, src2Func, mixFunc, filterFunc, ampFunc];
		var params =[];
		funcs.do{|f|
			params = params.addAll(this.getParams(f));
		 };
		^params;
	}

	getParams{|func|
		var num = func.def.numArgs;
		^Array.rand(num, 0.0, 1.0);
	}

	createExampleFunc{|params|
		^{
			var mod1, mod2, mod3, mod4, src1, src2, mix, filter, amp;
			#mod1, params = this.getObject(mod1Func, params, nil);
			#mod2, params = this.getObject(mod2Func, params, nil);
			#mod3, params = this.getObject(mod3Func, params, nil);
			#mod4, params = this.getObject(mod4Func, params, nil);
			#src1, params = this.getObject(src1Func, params, [mod1]);
			#src2, params = this.getObject(src2Func, params, [mod2]);
			#mix, params = this.getObject(mixFunc, params, [src1, src2]);
			#filter, params = this.getObject(filterFunc, params, [mix, mod3]);
			#amp, params = this.getObject(ampFunc, params, [filter, mod4]);
			amp;
		};
	}

	dumpExample{|params|
			var mod1, mod2, mod3, mod4, src1, src2, mix, filter, amp;
			#mod1, params = this.getObject(mod1Func, params, nil);
		    ("Source 1 Modulator:"+mod1.class.name).postln;mod1.dumpArgs;
		    mod1Func.def.argNames.postln;
		    #src1, params = this.getObject(src1Func, params, [mod1]);
		     ("Source 1:"+src1.class.name).postln;src1.dumpArgs;
		    src1Func.def.argNames.postln;
			#mod2, params = this.getObject(mod2Func, params, nil);
			("Source 2 Modulator:"+mod2.class.name).postln;mod2.dumpArgs;
		    #src2, params = this.getObject(src2Func, params, [mod2]);
		    ("Source 2:"+src2.class.name).postln;src2.dumpArgs;
		    #mix, params = this.getObject(mixFunc, params, [src1, src2]);
		    #mod3, params = this.getObject(mod3Func, params, nil);
		    ("Mix :").postln;mix.dumpArgs;
		    ("Filter Modulator:"+mod3.class.name).postln;mod3.dumpArgs;
		    #filter, params = this.getObject(filterFunc, params, [mix, mod3]);
		    ("Filter:"+filter.class.name).postln;filter.dumpArgs;
			#mod4, params = this.getObject(mod4Func, params, nil);
		    ("Amp Modulator:"+mod4.class.name).postln;mod4.dumpArgs;
			#amp, params = this.getObject(ampFunc, params, [filter, mod4]);
			amp!2;
	}



	getObject{|func, params, inputs|
		var obj;
		var n = func.def.numArgs;
		^[func.valueArray(params[..n-1]).valueArray(inputs), params[n..]];
	}


	addModulator{|class|
		^class.switch(
			DC, {
				{DC.kr(0)}
			},
			LFNoise2, {
				{|f| LFNoise2.kr(ControlSpec(2, 20, \exp, 0.1, 5, "Hz").map(f))}
			},
			LFTri, {
				{|f| LFTri.kr(ControlSpec(0.1, 20, \exp, 0.1, 5, "Hz").map(f))}
			},
			EnvGen, {
				{|a, d, s, r|
					var aSpec = ControlSpec(0.001, 0.5, \exp);
					var dSpec = ControlSpec(0.1, 1.0, \exp);
					var sSpec = ControlSpec(0.1, 0.9, \lin);
					var rSpec = ControlSpec(0.1, 1.0, \exp);
					2 * EnvGen.kr(Env.adsr(
						aSpec.map(a),
						dSpec.map(d),
						sSpec.map(s),
						rSpec.map(r)
					),
					ToggleFF.kr(Impulse.kr(0)+TDelay.kr(Impulse.kr(0), 3)) //3 secs sustain
					) - 1// map to -1,1
				}
			}
		)
	}


	addSource{|class|
		var modSpec=ControlSpec(0.0,1.0,3);
		^class.switch(
			Pulse, {
				{|modAmnt, width| {|mod| Pulse.ar(pitch, 0.5 +modAmnt*mod.range(-0.5, 0.5))}}
			},
			VarSaw, {
				{|modAmnt, width|{|mod| VarSaw.ar(pitch, 0.5 +modAmnt*mod.range(-0.5, 0.5))}}
			},
			Saw, {
				{|modAmnt|{|mod| Saw.ar(pitch *(2**(modSpec.map(modAmnt) * mod.range(-3, 3)) ))}}
			},

			WhiteNoise, {
				{|modAmnt| {|mod| WhiteNoise.ar(0.5 +modAmnt*mod.range(-0.5, 0.5))}}
			}
		)
	}

	addMix{
		^{|v|
			{|x1,x2|
				(v * x1.value) + ((1-v) * x2.value);
			}
		}
	}

	addFilter{|class|
		var modSpec=ControlSpec(0.0,1.0,3);
		var freqSpec = ControlSpec(60, 2000, \exp);
		var resSpec = ControlSpec(0.1, 0.9, \lin);

		^{|f, modAmnt, rq| {|mix, mod|
			class.ar(mix, freqSpec.map(f)* (2**(modSpec.map(modAmnt) * mod.range(-2, 2))), resSpec.map(rq))
		    }
		}
	}

	addAmp{
		var modSpec=ControlSpec(0.0,1.0,3);
		^{|modAmnt|
			{|signal, mod|
				signal *(modAmnt*mod.range(0, 1));
			}
		}
	}
}