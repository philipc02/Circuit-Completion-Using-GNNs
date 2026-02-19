spice
* SPICE Netlist
* Transistors
Q1 3 2 4 NPN_MODEL
Q2 2 3 4 NPN_MODEL
Q3 2 2 6 PNP_MODEL
Q4 2 2 3 NPN_MODEL
Q5 2 6 2 PNP_MODEL
Q6 4 7 2 NPN_MODEL

* Current Source
I1 4 3 DC I

* Voltage Sources
VCC 2 1 DC VCC_VALUE
VEE 4 5 DC VEE_VALUE
VI 6 0 AC VI_VALUE
IB 6 2 DC IB_VALUE

* Models (Assuming standard models)
.model NPN_MODEL NPN
.model PNP_MODEL PNP

.END