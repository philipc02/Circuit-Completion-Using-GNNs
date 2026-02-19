spice
* SPICE Netlist
V1 1 6 DC

R1 2 1 10k
R2 2 3 100k
R3 3 4 10k
R4 4 5 50k

* Op-Amps
* Ideal Op-amp model
.subckt OPAMP INP INM OUT VCC VEE
* Place the appropriate model or parameters for simulation
* This is a simplified representation
.ends OPAMP

* Connect the Op-Amps
XU1 2 1 3 OPAMP
XU2 3 1 4 OPAMP

* Voltage source connected to node 2
v_v1 2 7 DC 0

* Ground
V0 6 1 DC 0