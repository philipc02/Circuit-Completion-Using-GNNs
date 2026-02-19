* Nodes:
* 1 = Vb1
* 2 = VDD
* 3 = Vout
* 4 = Vin1
* 5 = Vin2
* 6 = Ground


M1 3 4 6 6 NMOS
M2 3 5 6 6 NMOS
M3 3 1 2 2 PMOS
M4 2 2 3 3 PMOS

I1 6 0 DC ISS

V1 2 0 DC VDD

* End of Netlist