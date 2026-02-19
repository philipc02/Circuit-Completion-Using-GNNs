spice
* SPICE Netlist for the given schematic

V_CC 6 0 DC VCC_VALUE

RB 5 4 RB_VALUE
RC 6 2 RC_VALUE

Q1 2 4 3 NPN

* Define the NPN model if not pre-defined
.model NPN NPN (IS=1e-14 BF=100 NF=1)

* Establish node connections
* - Node 6 is VCC
* - Node 2 is collector
* - Node 4 is base
* - Node 3 is emitter which is commonly considered grounded here