spice
* SPICE Netlist
Vcc 4 0 DC 10V
Vee 2 0 DC -10V
Vs 3 0 DC

R1 4 5 320k
R2 5 2 280k
R5 3 5 0.1k
RE 4 6 10k
RC 7 8 5k
RL 7 0 2k

CC1 5 6
CC2 8 7
CE 6 0

Q1 6 5 8 QNPN

.model QNPN NPN (Is=1e-14 bf=100)

* Connections
* 1 - Collector
* 2 - Base
* 3 - Emitter