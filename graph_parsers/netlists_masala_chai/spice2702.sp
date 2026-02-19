spice
* SPICE Netlist

M1 3 2 0 0 NMOS
M2 5 2 3 3 PMOS
MB 4 4 0 0 NMOS

IG 2 0 DC 0
IB1 4 0 DC 0

CG 2 2 0.01u
CB 2 3 0.01u

RG 2 3 1k

VDD 5 0 DC 5V

* Connections:
* - Node 0: Ground
* - Node 2: N
* - Node 3: Vout
* - Node 4: Vin, IB1
* - Node 5: VDD

.END