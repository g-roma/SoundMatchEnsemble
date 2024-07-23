(
// called from python to generate datasets for many synths
//var synthID = thisProcess.argv[0];
//var outputDir = thisProcess.argv[1];
//var numExamples = thisProcess.argv[2].asInteger;

// Test or run from SC
var synthID = "1111111";
var outputDir = "/path/to/1111111";
var numExamples = 5000;

var synth = SynthFactory.new(synthID);
Server.local.options.maxSynthDefs = 1000000;
Server.local.options.maxNodes = 1000000;
Server.local.options.memSize = 200000;
Server.local.options.numBuffers = 1000000;

File.mkdir(outputDir);
File.mkdir(outputDir++"/wav");
File.mkdir(outputDir++"/labels");

Server.local.waitForBoot{
	var remaining = numExamples;
	fork{
		numExamples.do{|r, i|
			var cond, params, b;
			params = synth.randomParams;
			b = synth.createExampleFunc(params)
			.asBuffer(4, s, action:{
				fork{
					b.write(outputDir++"/wav/"++i++".wav", "wav", "int16");
					File.use(outputDir++"/labels/"++i++".csv", "wt", { |file|
						params.do({|p|
							file.write(p.asString++",");
						});
						file.write("\n");
					});
					b.free;
					i.postln;
					remaining = remaining -1;
					if(remaining == 0){
						"done".postln;
						//5.wait;0.exit
					};
				}
			});
		0.02.wait;
		}
	}
}

)
