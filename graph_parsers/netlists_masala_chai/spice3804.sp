plaintext
* SPICE Netlist

VDD 3 0 DC 10V
VI 4 0 SIN(0 1V 1kHz)

* Resistors
R1 3 6 1k
R2 6 0 1k
RS 3 8 1k
Ro 8 5 1k
RL 5 9 1k

* Capacitors
CC1 4 6 1uF
CC2 8 5 1uF

* NMOS Transistor
M1 8 6 2 2 NMOS

* Analysis
.END