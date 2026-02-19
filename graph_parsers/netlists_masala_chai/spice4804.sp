plaintext
* SPICE Netlist

* Voltage Sources
VCC 9 0 DC 30V
Vin 7 0 AC 100uV

* Resistors
RB 2 9 1Meg
RC 3 9 5k
RL 5 0 100k

* Capacitors
C1 2 4 <value1>
C2 5 8 <value2>

* Transistor
Q1 3 4 0 BJT

.model BJT NPN (BF=100)

* Net Connections
Vout 5 0

.END