plaintext
* SPICE Netlist

* Voltage Source
VGG 1 0 DC 5G
Vi 2 0 PULSE(0 1V 0 1n 1n 1u 2u)

* MOSFET NMOS
M1 2 2 0 0 NMOS_MODEL

* BJT NPN
Q2 3 4 2 NPN_MODEL

* Resistors
RD1 5 3 10k
RE2 5 3 10k
RL 4 0 1k
Rof 4 vo 10k

* DC Voltage
V+ 5 0 DC 15V

* Models
.model NMOS_MODEL NMOS (Level=1)
.model NPN_MODEL NPN (IS=1e-15 BF=100)

* Analysis
.TRAN 1ns 10us
.END