plaintext
* SPICE Netlist
* Transistors
M1 3 Vin1 Vout1 Vout1 NMOS
M2 4 Vin2 Vout2 Vout2 NMOS
M11 Vout1 2 VDD VDD PMOS
M12 Vout2 5 VDD VDD PMOS
M_REF 5 3 VDD VDD NMOS

* Current Source
I_REF 3 4 DC 0.3mA

* Voltage Source
VDD 4 0 DC <Voltage_Value>