spice
* SPICE Netlist for the Given Circuit

V_CC 9 0 DC 15V
V_IN 13 4 AC 1V

* Resistors
RC1 10 9 1k
RC2 9 7 1k
RB11 7 6 10k
RB21 6 4 10k
RE 2 11 200
RB22 2 5 5k
RT 13 12 50k
RL 8 0 10k

* Capacitors
C1 12 6 10uF
C2 4 8 10uF

* BJTs
Q1 10 6 2 BJT_NPN
Q2 7 2 5 BJT_NPN

.model BJT_NPN NPN (Is=1e-14 bf=100)

.ends