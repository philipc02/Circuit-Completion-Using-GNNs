spice
* SPICE Netlist

* Voltage Sources
Vplus 7 0 DC 10
Vinput 5 0 DC 0

* Current Sources
I1 6 0 DC 1mA
I2 2 3 DC 2mA

* Resistors
RC2 7 2 18.6k
RC3 2 1 2k
RF 5 4 10k
RL 2 6 1k

* NPN Transistors
Q1 5 5 8 NPN
Q2 0 4 2 NPN

* PNP Transistor
Q3 2 2 1 PNP

* .MODEL Statements (default)
.model NPN NPN
.model PNP PNP

* Analysis
.control
tran 1us 100us
plot v(5) v(2) v(7)
.endc
.end