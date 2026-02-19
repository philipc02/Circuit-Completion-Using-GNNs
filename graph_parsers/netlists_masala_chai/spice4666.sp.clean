spice
* SPICE Netlist
Vin vin 1 DC 0
Vdc vdc 0 2 DC Vdc_value

RS vin 1 RS_value
R1 1 2 R1_value
R2 2 0 R2_value

D1 1 0 D

* Specify model for the diode
.model D D

* Control statements
*.dc Vdc 0 10 0.1
*.op
*.end