spice
* SPICE netlist for the given schematic

* Current Sources
I1 5 9 DC i_f
I2 5 11 DC i_ia^2
I3 2 5 DC 0

* Resistors
R1 5 10 5.5k
R2 8 7 20k
R3 4 3 5k
R4 4 3 500

* Transistors
Q1 5 7 6 NPN
Q2 2 3 4 NPN

* Node Annotation
* 1: Not used in the netlist, refer to node 7 directly
* 2: Base of Q2
* 3: Collector of Q2
* 4: Emitter of Q2
* 5: Common node for current sources and input to Q1
* 6: Emitter of Q1
* 7: Collector of Q1 and base of Q2
* 8: Base of Q1
* 9: Ground node for I1
* 10: Ground node for R1
* 11: Ground node for I2

.end