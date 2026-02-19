plaintext
* Differential Amplifier SPICE Netlist

VCC 6 0 DC 15V
VEE 3 0 DC -15V
VIN 1 0 DC 1mV

Q1 2 1 3 npn
Q2 5 4 3 npn

RC 4 5 1MEG
RE 2 3 1MEG

.model npn NPN

.control
run
.endc