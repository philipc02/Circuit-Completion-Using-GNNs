spice
* SPICE Netlist for the given circuit
VSD 1 0 DC <value_of_Vsd>
VOD 5 4 DC <value_of_Vod>

R1 1 3 <value_of_R1>
R2 1 2 <value_of_R2>
R3 3 5 <value_of_R3>
R4 2 4 <value_of_R4>

* Op-Amp model (ideal)
* .model OPAMP ideal
*.subckt opamp 3 2 5
E1 5 0 3 2 100k
*.ends opamp