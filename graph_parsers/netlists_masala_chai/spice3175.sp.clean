* SPICE Netlist

* Resistors
R1 X 2 R1_value
R2 Y 2 R2_value
R3 2 nA R3_value

* Voltage Source
VOS Y 2 DC VOS_value

* NMOS Transistors
M1 2 X 5 5 NMOS_model
M2 nA Y 4 4 NMOS_model

* Op-Amp
* Assuming ideal op-amp model, input terminals at nodes 2 (+) and Y (-)
* Output connected to Vout

* Output Voltage
Vout 2 0 DC 0

* Define models for NMOS (this is needed for simulation, provide actual parameters)
.model NMOS_model NMOS (LEVEL=1)

.end