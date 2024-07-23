// Using SynthFactory

(
a = [0,0,0,0,0,0,0].join;
b = SynthFactory.new(a);
x = b.randomParams;
x.postln;
y = b.createExampleFunc(x);
y.play(b.server)
)

// Using MonolithicSynth
(
m = MonolithicSynth.new;
y = m.randomParams;
z = m.createExampleFunc(y);
z.play
)

