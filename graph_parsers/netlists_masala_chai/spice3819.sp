* SPICE Netlist

* Transistor Definitions
* PMOS: M1 drain gate source bulk model length width
MD 4 2 3 3 PMOS_MODEL L=W=YOUR_PMOS_DIMENSIONS

* NMOS: M2 drain gate source bulk model length width
ML 4 5 0 0 NMOS_MODEL L=W=YOUR_NMOS_DIMENSIONS

* Voltage Source
Vplus 3 0 DC 2.5V

* Model Definitions
.model PMOS_MODEL PMOS
.model NMOS_MODEL NMOS

* End of Netlist