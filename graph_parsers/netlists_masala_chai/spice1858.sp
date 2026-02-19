* NMOS transistor: M1
M1 Drain_M1 Gate_M1 Source_M1 Source_M1 NMOS
* PMOS transistor: M2
M2 Drain_M2 Gate_M2 Source_M2 Source_M2 PMOS

* Resistors
RD Drain_M2 VDD  RD_VALUE
RF Gate_M2 Vout RF_VALUE

* Voltage Source
VDD VDD 0 DC VDD_VALUE

* Current Source
Iin Source_M1 0 DC Iin_VALUE

* Connections
Drain_M1 3
Gate_M1 Vb
Source_M1 4

Drain_M2 Vout
Gate_M2 3
Source_M2 2

Vout 0