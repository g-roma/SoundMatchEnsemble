(
// call from python to generate audio files for a parameter sets csv file
// using SynthFactory

var synthID = thisProcess.argv[0];
var filename = thisProcess.argv[1];
var output = thisProcess.argv[2];

var results = CSVFileReader.read(filename);
var synth = SynthFactory.new(synthID);
results.postln;
File.mkdir(output);
fork{
			Server.local.waitForBoot{
		           results.do{|r, i|
					    var c = Condition(false);
					    var params = r.collect{|x|x.asFloat};
					    var b = synth.createExampleFunc(params)
					        .asBuffer(4, s, action:{c.test=true;c.signal});
					    c.wait;
			            b.write(output++"/"++i++".wav", "wav", "int16");
					    s.sync;
			            i.postln;
				   };
		           0.exit;
			}
	}
)



