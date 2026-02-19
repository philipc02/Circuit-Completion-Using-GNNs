spice
* SPICE Netlist for the provided schematic

VCC 3 0 DC <Vcc_value>

* Resistors
R1 3 3 <R_value>
R2 10 0 <R_value>

* Diodes
D5 3 5 D_model
D6 3 8 D_model

* NPN Transistors
Q1 7 6 0 QNModel
Q2 10 9 0 QNModel

* PNP Transistors
Q3 5 6 3 QPModel
Q4 8 9 3 QPModel

* Capacitors
C1 4 22 <C_value>

* Current Sources
I1 7 0 <I1_value>
IB1 6 4 <IB_value>
IB2 9 22 <IB_value>

* Voltage Input
Vin 7 0 AC 1

.model QNModel NPN
.model QPModel PNP
.model D_model D

.end