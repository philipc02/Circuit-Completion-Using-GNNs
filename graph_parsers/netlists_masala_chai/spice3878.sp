spice
* SPICE Netlist
* Components
Q1 2 0 1 NPN
RB 0 1 25k
RC 2 4 1k
IQ 3 4 DC 0.5mA

* Voltage Sources
VCC 3 0 DC 5V
VEE 2 0 DC -5V
VB 0 1 DC 0V
VC 4 0 DC -1V

* Nodes
* 0: Ground
* 1: Base of Transistor Q1
* 2: Collector of Transistor Q1
* 3: Positive Terminal of VCC
* 4: Connects to Resistor RC and IQ

.end