spice
* SPICE Netlist for the Given Circuit

M1 3 1 4 4 NMOS  * NMOS with D=3, G=1, S=B=4
M2 3 2 2 2 PMOS  * PMOS with D=3, G=2, S=B=2

VDD 2 0 DC 5V    * DC Voltage Source for VDD
VIN 1 0 DC 0V    * Input Voltage Source for Vin

* Note: NMOS and PMOS models need to be defined separately.