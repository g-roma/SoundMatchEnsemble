(
// call from python to generate audio files for a parameter sets csv file
// using MonolithicSynth

var filename = thisProcess.argv[0];
var output = thisProcess.argv[1];
var results = CSVFileReader.read(filename);
var first = results[0].collect{|x|x.asFloat};
var synth = MonolithicSynth.new();
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



