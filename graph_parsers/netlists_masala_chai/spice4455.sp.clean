plaintext
* SPICE Netlist for given schematic
.include 0V_vcc_neg.inc
.include 0V_gnd.inc

* Transistors
Q1 net2 net7 net9 0V_vcc_neg NPN
Q2 net3 net7 net8 0V_vcc_neg NPN
Q3 net4 net6 net9 0V_gnd NPN
Q4 net5 net6 net8 0V_gnd NPN
Q5 net6 net3 net3 0V_gnd NPN
Q6 net7 net4 net4 0V_gnd NPN
Q7 net7 net6 net8 0V_gnd PNP
Q8 net27 net9 net10 0V_vcc_neg NPN
Q9 net22 net8 net2 0V_gnd PNP
Q10 net22 net9 net3 0V_gnd NPN
Q11 net21 net10 net4 0V_vcc_neg PNP
Q12 net4 net21 net21 0V_gnd NPN

* Current Sources
I1 net1 net27 DC 2mA
I2 net2 net22 DC 1mA
I3 net3 net21 DC 0.5mA

* Capacitors
C1 net44 net3 30p16F

* Resistors
R1 net6 net4 50k
R2 net6 net3 1.5k
R3 net27 net9 40k
R4 net10 net44 300

* Power Supplies
V+ net27 0V_vcc_neg DC 15V
V- net44 0V_gnd DC -15V

* Analysis
.TRAN 1ns 10ms
.end