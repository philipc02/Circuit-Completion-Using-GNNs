spice
* NMOS Inverter Circuit
M1 7 6 3 3 NMOS
R1 2 7 RD
VDD 2 0 DC V_DD
Vi 5 0 DC V_i
* Connections
* 4 is the intermediate node between RD and M1 drain
Vout 4 0 DC 0
* Simulation Commands
* .model statement and analysis need to be added based on device and simulation specifics
.ends