plaintext
* SPICE Netlist for the Circuit

* Voltage Sources
Vcc 8 0 DC 10
Vg 9 0 AC 1m

* Resistors
RG 9 1 600
R1 1 4 10k
R2 2 4 2.2k
RC1 5 8 3.6k
RC2 3 8 3.6k
RE1 2 4 1k
RE2 2 4 1k
RL 3 4 10k

* Capacitors
C1 9 4 
C2 5 6 
C3 3 6 

* Transistors
Q1 5 1 2 NPN
Q2 3 6 2 NPN

* Define Transistor Model
.model NPN npn (Is=1e-14 Bf=100)

* Analysis
.ac dec 10 10 1Meg
.end