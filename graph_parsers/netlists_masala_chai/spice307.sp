* Circuit Netlist

*Current Source
I1 9 11 DC ?

* Resistors
R_RS 10 11 RS
R_RPI 5 2 rpi
R_RB 5 4 rb
R_RL 3 7 RL

* Capacitors
C_CPI 5 6 Cpi
C_CMU 6 7 Cmu

* Voltage-Dependent Current Source
G_GM 6 7 5 2 gm

* Node Definitions:
* 1 - Node between R_RS and rpi
* 2 - Node between rpi and r_b
* 3 - Node connected to R_L
* 4 - Ground
* 5 - Node connected to base of voltage-dependent current source and capacitors
* 6 - Node between Cpi and Cmu
* 7 - Node connected to R_L
* 8 - Node connected to emitter
* 9 - Node connected to current source input
* 10 - Node connected to R_S and input voltage VI
* 11 - Ground

.end