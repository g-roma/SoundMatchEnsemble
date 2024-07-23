(
// generate monolithic dataset (experiment 2)
var rootDir = "path/to/monolithic/dataset";
Server.local.options.maxSynthDefs = 1000000;
Server.local.options.maxNodes = 1000000;
Server.local.options.memSize = 200000;
Server.local.options.numBuffers = 1000000;
Server.local.waitForBoot{
fork{
		5.do{|o|
			var offset = o*1000;
			var synth = MonolithicSynth.new;
			var synthDir = rootDir;
			File.mkdir(synthDir++"/wav");
			File.mkdir(synthDir++"/labels");
			1000.do{|i|
					var params = synth.randomParams;
					var b = synth.createExampleFunc(params)
					.asBuffer(4, action:{|b|
					b.write(synthDir++"/wav/"++(offset+i)++".wav", "wav", "int16");
					File.use(synthDir++"/labels/"++(offset+i)++".csv", "wt", { |file|
				            params.do({|p|
							   file.write(p.asString++",");
					        });
						    file.write("\n");
					    });
		            });
					i.postln;
				};
		        20.wait;
		}
	}
}
)



