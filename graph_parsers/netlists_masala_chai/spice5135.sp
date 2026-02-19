spice
* SPICE netlist for the circuit

VCC 3 0 DC 15

* Resistors
R1 1 2 22k
R2 2 0 10k
RE 5 0 5.6k
RC 2 3 6.8k
RS 3 0 1k
R3 3 7 1k
RF 6 7 47k

* Capacitors
C1 1 0 10uF
C2 2 5 10uF
C3 5 0 10uF
C4 3 2 22uF
C5 7 0 10uF

* NPN Transistor
Q1 2 1 5 0 2N3904

* Op-Amp
XOP 6 2 3 0 741C

* Voltage Input
Vinput 1 0 AC 1

* End of netlist