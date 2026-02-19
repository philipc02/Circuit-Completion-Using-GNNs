spice
* Bipolar Junction Transistor Amplifier Circuit
VCC 5 6 DC 10V

RB1 3 5 10k
RB2 4 6 20k
RE  7 6 1k

CI  2 4 10uF
CO  3 2 10uF
CE  7 6 100uF

Q1 3 4 7 NPN

.model NPN NPN (IS=1E-14 BF=100)

* Connections
* 1: Node
* 2: V_IN
* 3: Collector of Q
* 4: Base of Q
* 5: VCC
* 6: Ground
* 7: Emitter of Q