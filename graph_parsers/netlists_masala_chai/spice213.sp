spice
* SPICE Netlist

* Voltage Sources
V1 4 0 DC 0
V2 6 0 DC 0

* Current Sources
I1 5 0 DC 0.001
I2 3 0 DC 0.002

* Resistors
R1 5 2 1k
R2 2 3 1k

* Operational Amplifier
* Connecting the non-inverting input to Vi node
* Connecting the inverting input to Vy node
* Output connected to Vo

XOPAMP 3 2 44 OPAMP_MODEL

* .MODEL statement for the Op-Amp as an ideal model
.model OPAMP_MODEL opamp