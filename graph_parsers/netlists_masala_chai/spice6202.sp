spice
* SPICE netlist for the given schematic

* Transistors
Q1 6 7 0 NPN
Q2 2 3 8 NPN

* Resistors
RC 2 9 2k
RE 8 3 1k
RF 4 6 1k

* Current Source
Is 7 6 DC 0.5mA

* Voltage sources
VCC 9 0 DC 15V
VEE 3 8 DC -15V

* Nodes
* 0: Ground
* 2: VCC
* 3: Connection to RE
* 4: Connection to RF
* 5: Output Vo
* 6: Base of Q1
* 7: Is input
* 8: Connection to -VEE
* 9: Collector of Q2

.end