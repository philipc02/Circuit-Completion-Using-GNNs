plaintext
* SPICE Netlist for the given schematic

* Voltage Source
VDD 4 0 DC <V_DD_value>

* Resistor
RD 4 3 <R_D_value>

* NMOS Transistor
* M<Name> <Drain> <Gate> <Source> <Body> <ModelName> <L=<value>> <W=<value>>
M1 3 6 7 7 NMOS

* Current Sources
IRD 4 3 DC <I_RD_value>
In1 3 0 DC <I_n1_value>

* Models (Add your model parameters here)
.model NMOS NMOS(Level=1)