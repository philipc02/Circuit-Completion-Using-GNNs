* NMOS Transistor M1
M1 X Vin 0 0 NMOS

* NMOS Transistor M2
M2 Iout X 0 0 NMOS

* Resistor RD
RD VDD X RD_value

* Voltage Source Vin
Vin Vin 0 DC Vin_value

* Voltage Supply VDD
VDD VDD 0 DC VDD_value

* Current Source Iout
Iout Iout 0 DC Iout_value

* Model Definitions
.model NMOS NMOS