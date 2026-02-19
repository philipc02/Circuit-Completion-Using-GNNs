spice
* SPICE Netlist
Vin 1 5 DC 0V AC 1V
RS 5 2 1k
R1 2 0 1k
R2 2 5 1k
R3 5 0 1k
RL 4 3 1k
VZ 1 0 D1N750A ; Assuming a standard Zener diode

* Op-Amp (ideal, requires specific model definitions not shown here)
XOP 2 5 OPAMP_MODEL

* NPN Transistor (generic model)
Q1 4 2 3 NPN

.model D1N750A D(IS=0.1E-14 BV=4.7 IBV=0.001 ISR=0.1E-14 NR=2) 
.model OPAMP_MODEL opamp
.model NPN NPN (IS=1.0E-14 BF=100)

.end