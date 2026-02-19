spice
*SPICE Netlist
* Components
V1 2 0 DC 5V
RR 2 1 Rvalue

* PMOS (Qref)
M1 2 1 1 1 PMOS L=Lref W=Wref

* NMOS (Q1)
M2 3 2 0 0 NMOS L=LR W=W1

* NMOS (Q2)
M3 4 3 0 0 NMOS L=LR W=W2

* NMOS (Q3)
M4 5 4 0 0 NMOS L=LR W=W3

* Current Sources
I1 2 3 DC (W1*J/R)
I2 3 4 DC (W2*J/R)
I3 4 5 DC (W3*J/R)
IR 1 2 DC IRvalue

* Model Definitions
.model PMOS PMOS
.model NMOS NMOS

*.END