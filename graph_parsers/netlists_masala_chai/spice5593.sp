spice
* SPICE Netlist

* Voltage Source
Vin 9 2 DC V_Id

* Resistors
R1a 1 2 R1
R1b 9 2 R1
R2a 2 3 R2
R2b 3 4 R2
R2c 3 7 R2
RG 2 7 RG

* Op-amp (ideal model for simplicity)
XU1 2 3 4 0 OPAMP

* Output load connected to ground
ROut 7 5 R2

* Direct ground connection
GND 5 0

* Node Voltage Labels (for reference)
* +vout = 7, -vout = 5, +vin = 9, -vin = 2

* End of netlist