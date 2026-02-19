spice
* SPICE Netlist for the Schematic

* Transistors
Q1 2 Vin1 0 NPN_model
Q2 2 Vin2 0 NPN_model
Q3 2 2 Vout NPN_model
Q4 2 2 Vout NPN_model

* Current Source
I_EE 0 3 DC IEE_value

* Resistor
RP 2 2 R_value

* Voltage Source
VCC 2 0 DC VCC_value

* End of Netlist