(
// generate dataset for 12 synths ensemble (experiment 2)
// use offset and num to run more than one time
// start with local server off
var offset = 0; // training example numbers start from offset
var num = 1000;
var rootDir = "/path/to/ensemble/dataset/";
Server.local.options.maxSynthDefs = 1000000;
Server.local.options.maxNodes = 1000000;
Server.local.options.memSize = 200000;
Server.local.options.numBuffers = 1000000;
Server.local.waitForBoot{
fork{
2.do{|s1|
	2.do{|s2|
		1.do{|m1|
			1.do{|m2|
				1.do{|m3|
					1.do{|m4|
						3.do{|f|
								var synthID = [s1, s2, 0, 2, 1, 1, f].join;
								var factory = SynthFactory.new(synthID);
								var synthDir = rootDir++synthID;
								synthID.postln;
								File.mkdir(synthDir);
								File.mkdir(synthDir++"/wav");
								File.mkdir(synthDir++"/labels");
									num.do{|i|
										var c = Condition(false);
										var params = factory.randomParams;
										var b = factory.createExampleFunc(params)
										.asBuffer(4, factory.server, action:{|buf|
											b.write(synthDir++"/wav/"++(offset+i)++".wav", "wav", "int16");
											File.use(synthDir++"/labels/"++(offset+i)++".csv", "wt", { |file|
													params.do({|p|
														file.write(p.asString++",");
													});
													file.write("\n");
												});
												b.free;
											});
										i.postln;
									};
									20.wait;
									"------".postln;
						};
					};
				};
			};
		};
	};
}
}
}
)



