plaintext
* SPICE Netlist for the Circuit

*.model npn NPN (IS=1e-14 BF=100)
*.model pnp PNP (IS=1e-14 BF=100)

VBB1 3 5 DC VBB/2
VBB2 5 2 DC VBB/2
VCC 7 0 DC VCC

QN 7 3 6 npn
QP 6 5 2 pnp

RL 4 0 RL

VIN 1 3 DC 0

* Connections
* 1 -> V_in
* 2 -> Ground (-V_CC)
* 3 -> Base of Q_N
* 4 -> Connected RL
* 5 -> Base of Q_P
* 6 -> Emitter/Common terminal
* 7 -> Collector of Q_N (V_CC)

.END