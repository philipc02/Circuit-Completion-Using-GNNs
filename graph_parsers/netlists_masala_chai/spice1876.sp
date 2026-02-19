spice
* SPICE Netlist
I1 2 0 DC 0 ; Current Source Iin
I2 5 2 DC 0 ; Current Source
V1 5 0 DC Vcc ; Voltage Source Vcc
RF 2 3 1k ; Resistor RF

* PMOS Model: PMOS Q1
M1 2 22 6 6 PMOS ; PMOS: Drain Gate Source Body

* Model Definitions
.model PMOS PMOS (Level=1)

.end