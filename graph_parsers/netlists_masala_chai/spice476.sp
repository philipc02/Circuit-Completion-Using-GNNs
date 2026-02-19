plaintext
* SPICE Netlist

I1 7 0 DC I1_VALUE
I2 8 0 DC I2_VALUE
Cgs 4 8 Cgs_VALUE
G1 3 0 4 0 Gm_VALUE
Rd 3 5 Rd_VALUE

* Voltage and current source values
.param I1_VALUE=0.5
.param I2_VALUE=0.5
.param Cgs_VALUE=1e-12
.param Gm_VALUE=1e-3
.param Rd_VALUE=1k

* End of Netlist