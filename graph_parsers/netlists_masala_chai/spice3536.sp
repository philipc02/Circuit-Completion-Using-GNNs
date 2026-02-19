spice
* Components and Nodes
V1 8 0 DC 10V
R1 8 3 8k
R2 3 0 2k
R3 4 8 5k
R4 3 6 1.4k
C1 1 3 C1_value
CE 6 5 CE_value
Q1 4 3 6 NPN

* Define Parameters
.model NPN NPN (BF=200)

* Voltage and Connections
Vin 1 0

* Output
Vout 5 0

* End of Netlist
.end