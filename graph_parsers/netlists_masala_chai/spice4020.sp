spice
* SPICE Netlist for the Given Circuit

Vs 5 0 DC

C1 5 5

Iq 6 0 DC 1mA

* Assuming NMOS with Node 5 at Gate, 1 at Drain, and Grounded Source
M1 1 5 0 0 NMOS

Rc 1 4 2k

C2 1 2

Rl 2 3 10k

* Operating Points
.dc V1 -5 5

* Descriptions
* NMOS assumed body connected to source

.END