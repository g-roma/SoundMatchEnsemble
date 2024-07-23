MonolithicSynth{
	/*
	This class implements the same template asSynthFactory, but as an integrated synthesizer.
	Different modules are selected as parameters (e.g. src1Index 0 to 1 is Pulse 1 to 2 is VarSaw)
	Modulators are haerdwired.
	This is equivalent to 12 of the synthesizers that can be generated with SynthFactory
	*/

	var <server;
	var func, paramNames, specs;
	var pitch;
	var dur = 1.0;
	var numParams;

	*new {
		^super.new.init();
	}

	init{
		numParams = 19;
		pitch = 60.midicps;
	}

	randomParams{
		^Array.rand(numParams, 0.0, 1.0);
	}

	createExampleFunc{|params|
			var src1Index = (params[0]*1.999).floor;
		    var src2Index = (params[1]*1.999).floor;
		    var fIndex = (params[2]*1.999).floor;
			var aSpec = ControlSpec(0.001, 0.5, \exp);
			var dSpec = ControlSpec(0.1, 1.0, \exp);
			var sSpec = ControlSpec(0.1, 0.9, \lin);
			var rSpec = ControlSpec(0.1, 1.0, \exp);
		    var freqSpec = ControlSpec(60, 2000, \exp);
		    var resSpec = ControlSpec(0.1, 0.9, \lin);

		    var modSpec=ControlSpec(0.0,1.0,3);
			^{
				var mod1 =  LFTri.kr(ControlSpec(0.1, 20, \exp, 0.1, 5, "Hz").map(params[3]));
				var mod2 = DC.kr(0);
				var mod3 = 	2 * EnvGen.kr(Env.adsr(
						aSpec.map(params[4]),
						dSpec.map(params[5]),
						sSpec.map(params[6]),
						rSpec.map(params[7])
					),
					ToggleFF.kr(Impulse.kr(0)+TDelay.kr(Impulse.kr(0), 3))//3 secs sustain
					) - 1;// map to -1,1

				var mod4 = 	2 * EnvGen.kr(Env.adsr(
						aSpec.map(params[8]),
						dSpec.map(params[9]),
						sSpec.map(params[10]),
						rSpec.map(params[11])
					),
					ToggleFF.kr(Impulse.kr(0)+TDelay.kr(Impulse.kr(0), 3))//3 secs sustain
					)-1;// map to -1,1


				var src1 = Select.ar(src1Index, [
					Pulse.ar(pitch, 0.5 +params[12]*mod1.range(-0.5, 0.5)),
					VarSaw.ar(pitch, 0.5 +params[12]*mod1.range(-0.5, 0.5))
				]);
				var src2 = Select.ar(src1Index, [
					Saw.ar(pitch *(2**(modSpec.map(params[13]) * mod2.range(-3, 3)) )),
					WhiteNoise.ar(0.5 +params[13]*mod2.range(-0.5, 0.5))
				]);
				var mix = (params[14]*src1) + ((1-params[14])*src2);

			var filter =  Select.ar(fIndex, [
				RLPF.ar(mix, freqSpec.map(
					params[15])* (2**(modSpec.map(params[16]) * mod3.range(-2, 2))), resSpec.map(params[17])),
				RHPF.ar(mix, freqSpec.map(
					params[15])* (2**(modSpec.map(params[16]) * mod3.range(-2, 2))), resSpec.map(params[17])),
				BPF.ar(mix, freqSpec.map(
					params[15])* (2**(modSpec.map(params[16]) * mod3.range(-2, 2))), resSpec.map(params[17])),
			]);

			var output = filter * params[18] * mod4.range(0,1);
			output;
			}
	}
}