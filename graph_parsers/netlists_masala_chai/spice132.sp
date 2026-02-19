* SPICE Netlist for Given Circuit

* Voltage node definitions:
* 3 - v_o
* 2 - Ground Connection

* Dependent Current Sources
I1 2 3 GMB2
I2 2 2 GM

* Resistors
R1 2 5 R1
R2 3 2 R2

* Ground
VSS 2 0 DC 0

* .PARAM declarations for transconductance gain
.PARAM GMB2 = -gmb2
.PARAM GM = gm

* Analysis
.TRAN 1n 10u
.END