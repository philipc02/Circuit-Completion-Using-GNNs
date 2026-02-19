plaintext
* Transistors
M1 Vout Vin1 VDD VDD PMOS
M2 Vout Vin1 0 0 NMOS

* Voltage supply
VDD VDD 0 DC <voltage value>

* Inputs
Vin1 Vin1 0 DC <input voltage value>

* Output
Vout Vout 0

* Model Definitions
.model PMOS PMOS (Level=1)
.model NMOS NMOS (Level=1)

* End of Netlist